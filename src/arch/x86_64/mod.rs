#[macro_use]
mod regs;

// Codes in this module come mainly from https://github.com/rcore-os/RVM-Tutorial

mod ept;
mod memory;
mod msr;
mod vmx;
mod percpu;

use core::marker::PhantomData;

use crate::{GuestPageTableTrait, HyperCraftHal, VmCpus, HyperResult, vcpus, HyperError, hal::{PerCpuDevices, PerVmDevices}};
use bit_set::BitSet;
use page_table::PagingIf;
#[cfg(feature = "type1_5")]
pub use vmx::LinuxContext;

const VM_EXIT_INSTR_LEN_VMCALL: u8 = 3;

/// Initialize the hypervisor runtime.
pub fn init_hv_runtime() {
    if !vmx::has_hardware_support() {
        panic!("VMX not supported");
    }
}

/// Nested page table define.
pub use ept::ExtendedPageTable as NestedPageTable;

/// VCpu define.
pub use vmx::VmxVcpu as VCpu;
pub use percpu::PerCpu;
pub use vmx::{VmxExitReason, VmxExitInfo};

// pub use device::{Devices, PortIoDevice};

////// Following are things to be implemented

/// VM define.
pub struct VM<H: HyperCraftHal, PD: PerCpuDevices<H>, VD: PerVmDevices<H>> {
    vcpus: VmCpus<H, PD>,
    vcpu_bond: BitSet,
    device: VD,
}

impl<H: HyperCraftHal, PD: PerCpuDevices<H>, VD: PerVmDevices<H>> VM<H, PD, VD> {
    /// Create a new [`VM`].
    pub fn new(vcpus: VmCpus<H, PD>) -> Self {
        Self { vcpus, vcpu_bond: BitSet::new(), device: VD::new().unwrap() }
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
                },
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
                    info!("{:#x?}", regs);
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
                    let result = vcpu_device.vmexit_handler(vcpu, &exit_info)
                        .or_else(|| self.device.vmexit_handler(vcpu, &exit_info));

                    match result {
                        Some(result) => {
                            if result.is_err() {
                                panic!("VM failed to handle a vm-exit: {:?}, error {:?}, vcpu: {:#x?}", exit_info.exit_reason, result.unwrap_err(), vcpu);
                            }
                        },
                        None => {
                            panic!("nobody wants to handle this vm-exit: {:?}, vcpu: {:#x?}", exit_info, vcpu);
                        },
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
        use memory_addr::PhysAddr;

        let (vcpu, vcpu_device) = self.vcpus.get_vcpu_and_device(vcpu_id).unwrap();
        loop {
            if let Some(exit_info) = vcpu.run_type15(linux) {
                
                let rip = exit_info.guest_rip as *const u8;
                let length = exit_info.exit_instruction_length as usize;
                debug!("rip:{:#x} length:{:#x}", exit_info.guest_rip, length);
                // Create a slice from the host address.
                let instruction_slice =
                    unsafe { core::slice::from_raw_parts(rip, length) };

                // Print the instruction.
                for byte in instruction_slice {
                    debug!("{:02x} ", byte);
                }
                debug!("{:#?}", instruction_slice);
        
                if exit_info.exit_reason == VmxExitReason::VMCALL {
                    let regs = vcpu.regs();
                    let id = regs.rax as u32;
                    let args = (regs.rdi as usize, regs.rsi as usize, regs.rdx as usize);
                    
                    info!("{:#x?}", regs);
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
                    let result = vcpu_device.vmexit_handler(vcpu, &exit_info);
                    debug!("this is result {:?}", result);
                    match result {
                        Some(result) => {
                            if result.is_err() {
                                panic!("VM failed to handle a vm-exit: {:?}, error {:?}, vcpu: {:#x?}", exit_info.exit_reason, result.unwrap_err(), vcpu);
                            }
                        },
                        None => {
                            panic!("nobody wants to handle this vm-exit: {:?}, vcpu: {:#x?}", exit_info, vcpu);
                        },
                    }
                }
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
                },
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
}

/// VM exit information.
pub use VmxExitInfo as VmExitInfo;

/// General purpose register index.
pub enum GprIndex {}

/// Hypercall message.
pub enum HyperCallMsg {}

