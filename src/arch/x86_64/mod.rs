#[macro_use]
mod regs;

// Codes in this module come mainly from https://github.com/rcore-os/RVM-Tutorial

mod ept;
mod memory;
mod msr;
mod percpu;
mod vmx;

use crate::{
    hal::{PerCpuDevices, PerVmDevices},
    vcpus, GuestPageTableTrait, GuestPhysAddr, GuestVirtAddr, HostPhysAddr, HostVirtAddr,
    HyperCraftHal, HyperError, HyperResult, VmCpus,
};
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use bit_set::BitSet;
use core::marker::PhantomData;
use iced_x86::{Decoder, DecoderOptions, Formatter, Instruction, MasmFormatter, OpKind};
use memory_addr::PhysAddr;
use page_table::PagingIf;
#[cfg(feature = "type1_5")]
pub use vmx::LinuxContext;
use x86_64::registers::debug;

const VM_EXIT_INSTR_LEN_VMCALL: u8 = 3;

/// Initialize the hypervisor runtime.
pub fn init_hv_runtime() {
    if !vmx::has_hardware_support() {
        panic!("VMX not supported");
    }
}

/// Nested page table define.
pub use ept::ExtendedPageTable as NestedPageTable;

pub use ept::GuestPageWalkInfo;
pub use percpu::PerCpu;
/// VCpu define.
pub use vmx::VmxVcpu as VCpu;
pub use vmx::{VmxExitInfo, VmxExitReason};

// pub use device::{Devices, PortIoDevice};

////// Following are things to be implemented

const PAGE_FAULT_ID_FLAG: u32 = 0x00000010;
const PAGE_FAULT_P_FLAG: u32 = 0x00000001;
const PAGE_ENTRY_CNT: usize = 512;
const PAGE_SIZE: usize = 0x1000;

/// VM define.
pub struct VM<H: HyperCraftHal, PD: PerCpuDevices<H>, VD: PerVmDevices<H>, G: GuestPageTableTrait> {
    vcpus: VmCpus<H, PD>,
    vcpu_bond: BitSet,
    device: VD,
    /// EPT
    pub ept: Arc<G>,
}

impl<H: HyperCraftHal, PD: PerCpuDevices<H>, VD: PerVmDevices<H>, G: GuestPageTableTrait>
    VM<H, PD, VD, G>
{
    /// Create a new [`VM`].
    pub fn new(vcpus: VmCpus<H, PD>, ept: Arc<G>) -> Self {
        Self {
            vcpus,
            vcpu_bond: BitSet::new(),
            device: VD::new().unwrap(),
            ept,
        }
    }

    /// Bind the specified [`VCpu`] to current physical processor.
    pub fn bind_vcpu(&mut self, vcpu_id: usize) -> HyperResult<(&mut VCpu<H>, &mut PD)> {
        if self.vcpu_bond.contains(vcpu_id) {
            Err(HyperError::InvalidParam)
        } else {
            match self.vcpus.get_vcpu_and_device(vcpu_id) {
                Ok((vcpu, device)) => {
                    self.vcpu_bond.insert(vcpu_id);
                    vcpu.bind_to_current_processor()?;
                    Ok((vcpu, device))
                }
                e @ Err(_) => e,
            }
        }
    }

    #[allow(unreachable_code)]
    /// Run a specified [`VCpu`] on current logical vcpu.
    pub fn run_vcpu(&mut self, vcpu_id: usize) -> HyperResult {
        let (vcpu, vcpu_device) = self.vcpus.get_vcpu_and_device(vcpu_id).unwrap();

        loop {
            if let Some(exit_info) = vcpu.run() {
                // we need to handle vm-exit this by ourselves

                if exit_info.exit_reason == VmxExitReason::VMCALL {
                    let regs = vcpu.regs();
                    trace!("{:#x?}", regs);
                    let id = regs.rax as u32;
                    let args = (regs.rdi as usize, regs.rsi as usize, regs.rdx as usize);

                    match vcpu_device.hypercall_handler(vcpu, id, args) {
                        Ok(result) => vcpu.regs_mut().rax = result as u64,
                        Err(e) => panic!("Hypercall failed: {e:?}, hypercall id: {id:#x}, args: {args:#x?}, vcpu: {vcpu:#x?}"),
                    }

                    vcpu.advance_rip(VM_EXIT_INSTR_LEN_VMCALL)?;
                } else if exit_info.exit_reason == VmxExitReason::EXCEPTION_NMI {
                    match vcpu_device.nmi_handler(vcpu) {
                        Ok(result) => vcpu.regs_mut().rax = result as u64,
                        Err(e) => panic!("nmi_handler failed: {e:?}"),
                    }
                } else {
                    let guest_rip = exit_info.guest_rip;
                    let length = exit_info.exit_instruction_length;
                    let instr = Self::decode_instr(self.ept.clone(), vcpu, guest_rip, length)?;
                    let result = vcpu_device
                        .vmexit_handler(vcpu, &exit_info)
                        .or_else(|| self.device.vmexit_handler(vcpu, &exit_info, Some(instr)));

                    match result {
                        Some(result) => {
                            if result.is_err() {
                                panic!(
                                    "VM failed to handle a vm-exit: {:?}, error {:?}, vcpu: {:#x?}",
                                    exit_info.exit_reason,
                                    result.unwrap_err(),
                                    vcpu
                                );
                            }
                        }
                        None => {
                            panic!(
                                "nobody wants to handle this vm-exit: {:?}, vcpu: {:#x?}",
                                exit_info, vcpu
                            );
                        }
                    }
                }
            }

            vcpu_device.check_events(vcpu)?;
        }

        Ok(())
    }

    #[cfg(feature = "type1_5")]
    #[allow(unreachable_code)]
    /// Run a specified [`VCpu`] on current logical vcpu.
    pub fn run_type15_vcpu(&mut self, vcpu_id: usize, linux: &LinuxContext) -> HyperResult {
        let (vcpu, vcpu_device) = self.vcpus.get_vcpu_and_device(vcpu_id).unwrap();
        loop {
            if let Some(exit_info) = vcpu.run_type15(linux) {
                if exit_info.exit_reason == VmxExitReason::VMCALL {
                    let regs = vcpu.regs();
                    let id = regs.rax as u32;
                    let args = (regs.rdi as usize, regs.rsi as usize, regs.rdx as usize);

                    trace!("{:#x?}", regs);
                    match vcpu_device.hypercall_handler(vcpu, id, args) {
                        Ok(result) => vcpu.regs_mut().rax = result as u64,
                        Err(e) => panic!("Hypercall failed: {e:?}, hypercall id: {id:#x}, args: {args:#x?}, vcpu: {vcpu:#x?}"),
                    }

                    vcpu.advance_rip(VM_EXIT_INSTR_LEN_VMCALL)?;
                } else if exit_info.exit_reason == VmxExitReason::EXCEPTION_NMI {
                    match vcpu_device.nmi_handler(vcpu) {
                        Ok(result) => vcpu.regs_mut().rax = result as u64,
                        Err(e) => panic!("nmi_handler failed: {e:?}"),
                    }
                } else {
                    let guest_rip = exit_info.guest_rip;
                    let length = exit_info.exit_instruction_length;
                    let instr = Self::decode_instr(self.ept.clone(), vcpu, guest_rip, length)?;
                    let result = vcpu_device
                        .vmexit_handler(vcpu, &exit_info)
                        .or_else(|| self.device.vmexit_handler(vcpu, &exit_info, Some(instr)));
                    debug!("this is result {:?}", result);
                    match result {
                        Some(result) => {
                            if result.is_err() {
                                panic!(
                                    "VM failed to handle a vm-exit: {:?}, error {:?}, vcpu: {:#x?}",
                                    exit_info.exit_reason,
                                    result.unwrap_err(),
                                    vcpu
                                );
                            }
                        }
                        None => {
                            panic!(
                                "nobody wants to handle this vm-exit: {:?}, vcpu: {:#x?}",
                                exit_info, vcpu
                            );
                        }
                    }
                }
                debug!("test decode instruction");
                let guest_rip = exit_info.guest_rip;
                let length = exit_info.exit_instruction_length;
                let _instr = Self::decode_instr(self.ept.clone(), vcpu, guest_rip, length)?;
            }
            // vcpu_device.check_events(vcpu)?;
        }
    }

    /// Unbind the specified [`VCpu`] bond by [`VM::<H>::bind_vcpu`].
    pub fn unbind_vcpu(&mut self, vcpu_id: usize) -> HyperResult {
        if self.vcpu_bond.contains(vcpu_id) {
            match self.vcpus.get_vcpu_and_device(vcpu_id) {
                Ok((vcpu, _)) => {
                    self.vcpu_bond.remove(vcpu_id);
                    vcpu.unbind_from_current_processor()?;
                    Ok(())
                }
                Err(e) => Err(e),
            }
        } else {
            Err(HyperError::InvalidParam)
        }
    }

    /// Get per-vm devices.
    pub fn devices(&mut self) -> &mut VD {
        &mut self.device
    }

    /// Get vcpu and its devices by its id.
    pub fn get_vcpu_and_device(&mut self, vcpu_id: usize) -> HyperResult<(&mut VCpu<H>, &mut PD)> {
        self.vcpus.get_vcpu_and_device(vcpu_id)
    }

    /// decode guest instruction
    pub fn decode_instr(
        ept: Arc<G>,
        vcpu: &VCpu<H>,
        guest_rip: usize,
        length: u32,
    ) -> HyperResult<Instruction> {
        let asm = Self::get_gva_content_bytes(ept, guest_rip, length, vcpu)?;
        let asm_slice = asm.as_slice();
        // Only one isntruction
        let mut decoder = Decoder::with_ip(64, asm_slice, guest_rip as u64, DecoderOptions::NONE);
        let instr = decoder.decode();
        // print instruction
        let mut output = String::new();
        let mut formatter = MasmFormatter::new();
        formatter.format(&instr, &mut output);
        debug!("Instruction: {}", output);
        Ok(instr)
    }

    /// get gva content bytes
    pub fn get_gva_content_bytes(
        ept: Arc<G>,
        guest_rip: usize,
        length: u32,
        vcpu: &VCpu<H>,
    ) -> HyperResult<Vec<u8>> {
        debug!(
            "get_gva_content_bytes: guest_rip: {:#x}, length: {:#x}",
            guest_rip, length
        );
        let gva = vcpu.gla2gva(guest_rip);
        debug!("get_gva_content_bytes: gva: {:#x}", gva);
        let gpa = Self::gva2gpa(ept.clone(), vcpu, gva)?;
        debug!("get_gva_content_bytes: gpa: {:#x}", gpa);
        let hva = Self::gpa2hva(ept.clone(), gpa)?;
        debug!("get_gva_content_bytes: hva: {:#x}", hva);
        let mut content = Vec::with_capacity(length as usize);
        let code_ptr = hva as *const u8;
        unsafe {
            for i in 0..length {
                let value_ptr = code_ptr.offset(i as isize);
                content.push(value_ptr.read());
            }
        }
        debug!("get_gva_content_bytes: content: {:#?}", content);
        Ok(content)
    }

    fn gpa2hva(ept: Arc<G>, gpa: GuestPhysAddr) -> HyperResult<HostVirtAddr> {
        let hpa = Self::gpa2hpa(ept, gpa)?;
        let hva = H::phys_to_virt(hpa);
        Ok(hva as HostVirtAddr)
    }

    fn gpa2hpa(ept: Arc<G>, gpa: GuestPhysAddr) -> HyperResult<HostPhysAddr> {
        ept.translate(gpa)
    }

    fn gva2gpa(ept: Arc<G>, vcpu: &VCpu<H>, gva: GuestVirtAddr) -> HyperResult<GuestPhysAddr> {
        let guest_ptw_info = vcpu.get_ptw_info();
        Self::page_table_walk(ept, guest_ptw_info, gva)
    }

    // suppose it is 4-level page table
    fn page_table_walk(
        ept: Arc<G>,
        pw_info: GuestPageWalkInfo,
        gva: GuestVirtAddr,
    ) -> HyperResult<GuestPhysAddr> {
        debug!("page_table_walk: gva: {:#x} pw_info:{:#?}", gva, pw_info);
        if pw_info.level <= 1 {
            return Ok(gva as GuestPhysAddr);
        }
        let mut addr = pw_info.top_entry;
        let mut current_level = pw_info.level;
        let mut shift = 0;
        while current_level != 0 {
            current_level -= 1;
            // get page table base addr
            addr = addr & !(PAGE_ENTRY_CNT - 1);
            if current_level == 2 {
                let _a = 0;
            }
            let base = Self::gpa2hva(ept.clone(), addr)?;
            shift = (current_level * pw_info.width as usize) + 12;
            let index = (gva >> shift) & (PAGE_ENTRY_CNT - 1);
            // get page table entry pointer
            let entry_ptr = unsafe { (base as *const usize).offset(index as isize) };
            // next page table addr (gpa)
            addr = unsafe { *entry_ptr };
        }

        let mut entry = addr;
        // debug!("1 page_table_walk: entry: {:#x} shift:{:#x}", entry, shift);
        // ?????
        entry >>= shift;
        // debug!("2 page_table_walk: entry: {:#x} shift:{:#x}", entry, shift);
        /* shift left 12bit more and back to clear XD/Prot Key/Ignored bits */
        entry <<= shift + 12;
        // debug!("3 page_table_walk: entry: {:#x} shift:{:#x}", entry, shift);
        entry >>= 12;
        // debug!("4 page_table_walk: entry: {:#x} shift:{:#x}", entry, shift);
        Ok((entry | (gva & (PAGE_SIZE - 1))) as GuestPhysAddr)
    }
}

/// VM exit information.
pub use VmxExitInfo as VmExitInfo;

/// General purpose register index.
pub enum GprIndex {}

/// Hypercall message.
pub enum HyperCallMsg {}
