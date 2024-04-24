use alloc::collections::VecDeque;
use core::fmt::{Debug, Formatter, Result};
use core::{arch::asm, mem::size_of};
use x86::msr::IA32_EFER;
use x86_64::registers::debug;

use bit_field::BitField;
use raw_cpuid::CpuId;
use x86::bits64::vmx;
use x86::controlregs::{xcr0 as xcr0_read, xcr0_write, Xcr0};
use x86::dtables::{self, DescriptorTablePointer};
use x86::segmentation::SegmentSelector;
use x86_64::registers::control::{Cr0, Cr0Flags, Cr3, Cr4, Cr4Flags, EferFlags};

use super::definitions::VmxExitReason;
use super::region::{MsrBitmap, VmxRegion};
#[cfg(feature = "type1_5")]
use super::segmentation::Segment;
use super::vmcs::{
    self, VmcsControl32, VmcsControl64, VmcsControlNW, VmcsGuest16, VmcsGuest32, VmcsGuest64,
    VmcsGuestNW, VmcsHost16, VmcsHost32, VmcsHost64, VmcsHostNW,
};
#[cfg(feature = "type1_5")]
use super::LinuxContext;
use super::VmxPerCpuState;
use crate::arch::{
    ept::GuestPageWalkInfo, memory::NestedPageFaultInfo, msr::Msr, regs::GeneralRegisters,
};
use crate::{
    GuestPhysAddr, GuestVirtAddr, HostPhysAddr, HyperCraftHal, HyperError, HyperResult, VmxExitInfo,
};

#[derive(Clone)]
pub struct XState {
    host_xcr0: u64,
    guest_xcr0: u64,
    host_xss: u64,
    guest_xss: u64,
}

#[derive(PartialEq, Eq, Debug)]
pub enum VmCpuMode {
    Real,
    Protected,
    Compatibility, // IA-32E mode (CS.L = 0)
    Mode64,        // IA-32E mode (CS.L = 1)
}

impl XState {
    /// Create a new [`XState`] instance with current host state
    fn new() -> Self {
        let xcr0 = unsafe { xcr0_read().bits() };
        let xss = Msr::IA32_XSS.read();

        Self {
            host_xcr0: xcr0,
            guest_xcr0: xcr0,
            host_xss: xss,
            guest_xss: xss,
        }
    }

    fn enable_xsave() {
        unsafe { Cr4::write(Cr4::read() | Cr4Flags::OSXSAVE) };
    }
}

const MSR_IA32_EFER_LMA_BIT: u64 = 1 << 10;
const CR0_PE: usize = 1 << 0;

/// A virtual CPU within a guest.
#[repr(C)]
pub struct VmxVcpu<H: HyperCraftHal> {
    // DO NOT modify `guest_regs` and `host_stack_top` and their order unless you do know what you are doing!
    // DO NOT add anything before or between them unless you do know what you are doing!
    guest_regs: GeneralRegisters,
    host_stack_top: u64,
    vcpu_id: usize,
    launched: bool,
    vmcs: VmxRegion<H>,
    msr_bitmap: MsrBitmap<H>,
    pending_events: VecDeque<(u8, Option<u32>)>,
    xstate: XState,
}

impl<H: HyperCraftHal> VmxVcpu<H> {
    /// Create a new [`VmxVcpu`].
    pub fn new(
        vcpu_id: usize,
        vmcs_revision_id: u32,
        entry: GuestPhysAddr,
        ept_root: HostPhysAddr,
    ) -> HyperResult<Self> {
        XState::enable_xsave();
        let mut vcpu = Self {
            guest_regs: GeneralRegisters::default(),
            host_stack_top: 0,
            vcpu_id,
            launched: false,
            vmcs: VmxRegion::new(vmcs_revision_id, false)?,
            msr_bitmap: MsrBitmap::passthrough_all()?,
            pending_events: VecDeque::with_capacity(8),
            xstate: XState::new(),
        };
        vcpu.setup_msr_bitmap()?;
        vcpu.setup_vmcs(entry, ept_root)?;
        info!("[HV] created VmxVcpu(vmcs: {:#x})", vcpu.vmcs.phys_addr());
        Ok(vcpu)
    }

    /// Get the identifier of this [`VmxVcpu`].
    pub fn vcpu_id(&self) -> usize {
        self.vcpu_id
    }

    /// Bind this [`VmxVcpu`] to current logical processor.
    pub fn bind_to_current_processor(&self) -> HyperResult {
        unsafe {
            vmx::vmptrld(self.vmcs.phys_addr() as u64)?;
        }
        Ok(())
    }

    /// Unbind this [`VmxVcpu`] from current logical processor.
    pub fn unbind_from_current_processor(&self) -> HyperResult {
        unsafe {
            vmx::vmclear(self.vmcs.phys_addr() as u64)?;
        }
        Ok(())
    }

    /// Get CPU mode of the guest.
    pub fn get_cpu_mode(&self) -> VmCpuMode {
        let ia32_efer = Msr::IA32_EFER.read();
        let cs_access_right = VmcsGuest32::CS_ACCESS_RIGHTS.read().unwrap();
        let cr0 = VmcsGuestNW::CR0.read().unwrap();
        if (ia32_efer & MSR_IA32_EFER_LMA_BIT) != 0 {
            if (cs_access_right & 0x2000) != 0 {
                // CS.L = 1
                return VmCpuMode::Mode64;
            } else {
                return VmCpuMode::Compatibility;
            }
        } else if (cr0 & CR0_PE) != 0 {
            return VmCpuMode::Protected;
        } else {
            return VmCpuMode::Real;
        }
    }

    /// Run the guest. It returns when a vm-exit happens and returns the vm-exit if it cannot be handled by this [`VmxVcpu`] itself.
    pub fn run(&mut self) -> Option<VmxExitInfo> {
        // Inject pending events
        if self.launched {
            self.inject_pending_events().unwrap();
        }

        // Run guest
        self.load_guest_xstate();
        unsafe {
            if self.launched {
                self.vmx_resume();
            } else {
                self.launched = true;
                VmcsHostNW::RSP
                    .write(&self.host_stack_top as *const _ as usize)
                    .unwrap();

                self.vmx_launch();
            }
        }
        self.load_host_xstate();

        // Handle vm-exits
        let exit_info = self.exit_info().unwrap();
        trace!("VM exit: {:#x?}", exit_info);

        let cr4 = VmcsGuestNW::CR4.read().unwrap();
        if cr4.get_bit(18) {
            // panic!("osxsave dead!");
        }

        match self.builtin_vmexit_handler(&exit_info) {
            Some(result) => {
                if result.is_err() {
                    panic!("VmxVcpu failed to handle a VM-exit that should be handled by itself: {:?}, error {:?}, vcpu: {:#x?}", exit_info.exit_reason, result.unwrap_err(), self);
                }

                None
            }
            None => Some(exit_info),
        }
    }

    /// Basic information about VM exits.
    pub fn exit_info(&self) -> HyperResult<vmcs::VmxExitInfo> {
        vmcs::exit_info()
    }

    /// Information for VM exits due to external interrupts.
    pub fn interrupt_exit_info(&self) -> HyperResult<vmcs::VmxInterruptInfo> {
        vmcs::interrupt_exit_info()
    }

    /// Information for VM exits due to I/O instructions.
    pub fn io_exit_info(&self) -> HyperResult<vmcs::VmxIoExitInfo> {
        vmcs::io_exit_info()
    }

    /// Information for VM exits due to nested page table faults (EPT violation).
    pub fn nested_page_fault_info(&self) -> HyperResult<NestedPageFaultInfo> {
        vmcs::ept_violation_info()
    }

    /// Guest general-purpose registers.
    pub fn regs(&self) -> &GeneralRegisters {
        &self.guest_regs
    }

    /// Mutable reference of guest general-purpose registers.
    pub fn regs_mut(&mut self) -> &mut GeneralRegisters {
        &mut self.guest_regs
    }

    /// Guest stack pointer. (`RSP`)
    pub fn stack_pointer(&self) -> usize {
        VmcsGuestNW::RSP.read().unwrap()
    }

    /// Set guest stack pointer. (`RSP`)
    pub fn set_stack_pointer(&mut self, rsp: usize) {
        VmcsGuestNW::RSP.write(rsp).unwrap()
    }

    /// Translate guest virtual addr to linear addr    
    pub fn gla2gva(&self, guest_rip: usize) -> GuestVirtAddr {
        let cpu_mode = self.get_cpu_mode();
        let seg_base;
        if cpu_mode == VmCpuMode::Mode64 {
            seg_base = 0;
        } else {
            seg_base = VmcsGuestNW::CS_BASE.read().unwrap();
        }
        debug!(
            "seg_base: {:#x}, guest_rip: {:#x} cpu mode:{:?}",
            seg_base, guest_rip, cpu_mode
        );
        seg_base + guest_rip
    }

    /// Get Translate guest page table info
    pub fn get_ptw_info(&self) -> GuestPageWalkInfo {
        let top_entry = VmcsGuestNW::CR3.read().unwrap();
        let level = self.get_paging_level();
        let is_write_access = false;
        let is_inst_fetch = false;
        let is_user_mode_access = ((VmcsGuest32::SS_ACCESS_RIGHTS.read().unwrap() >> 5) & 0x3) == 3;
        let mut pse = true;
        let mut nxe =
            (VmcsGuest64::IA32_EFER.read().unwrap() & EferFlags::NO_EXECUTE_ENABLE.bits()) != 0;
        let wp = (VmcsGuestNW::CR0.read().unwrap() & Cr0Flags::WRITE_PROTECT.bits() as usize) != 0;
        let is_smap_on = (VmcsGuestNW::CR4.read().unwrap()
            & Cr4Flags::SUPERVISOR_MODE_ACCESS_PREVENTION.bits() as usize)
            != 0;
        let is_smep_on = (VmcsGuestNW::CR4.read().unwrap()
            & Cr4Flags::SUPERVISOR_MODE_EXECUTION_PROTECTION.bits() as usize)
            != 0;
        let width: u32;
        if level == 4 || level == 3 {
            width = 9;
        } else if level == 2 {
            width = 10;
            pse = VmcsGuestNW::CR4.read().unwrap() & Cr4Flags::PAGE_SIZE_EXTENSION.bits() as usize
                != 0;
            nxe = false;
        } else {
            width = 0;
        }
        GuestPageWalkInfo {
            top_entry,
            level,
            width: width,
            is_user_mode_access,
            is_write_access,
            is_inst_fetch,
            pse,
            wp,
            nxe,
            is_smap_on,
            is_smep_on,
        }
    }

    /// Guest rip. (`RIP`)
    pub fn rip(&self) -> usize {
        VmcsGuestNW::RIP.read().unwrap()
    }

    /// Guest cs. (`cs`)
    pub fn cs(&self) -> u16 {
        VmcsGuest16::CS_SELECTOR.read().unwrap()
    }

    /// Advance guest `RIP` by `instr_len` bytes.
    pub fn advance_rip(&mut self, instr_len: u8) -> HyperResult {
        Ok(VmcsGuestNW::RIP.write(VmcsGuestNW::RIP.read()? + instr_len as usize)?)
    }

    /// Add a virtual interrupt or exception to the pending events list,
    /// and try to inject it before later VM entries.
    pub fn queue_event(&mut self, vector: u8, err_code: Option<u32>) {
        self.pending_events.push_back((vector, err_code));
    }

    /// If enable, a VM exit occurs at the beginning of any instruction if
    /// `RFLAGS.IF` = 1 and there are no other blocking of interrupts.
    /// (see SDM, Vol. 3C, Section 24.4.2)
    pub fn set_interrupt_window(&mut self, enable: bool) -> HyperResult {
        let mut ctrl = VmcsControl32::PRIMARY_PROCBASED_EXEC_CONTROLS.read()?;
        let bits = vmcs::controls::PrimaryControls::INTERRUPT_WINDOW_EXITING.bits();
        if enable {
            ctrl |= bits
        } else {
            ctrl &= !bits
        }
        VmcsControl32::PRIMARY_PROCBASED_EXEC_CONTROLS.write(ctrl)?;
        Ok(())
    }
}

// Implementation of private methods
impl<H: HyperCraftHal> VmxVcpu<H> {
    fn setup_msr_bitmap(&mut self) -> HyperResult {
        // Intercept IA32_APIC_BASE MSR accesses
        let msr = x86::msr::IA32_APIC_BASE;
        self.msr_bitmap.set_read_intercept(msr, true);
        self.msr_bitmap.set_write_intercept(msr, true);
        // Intercept all x2APIC MSR accesses
        for msr in 0x800..=0x83f {
            self.msr_bitmap.set_read_intercept(msr, true);
            self.msr_bitmap.set_write_intercept(msr, true);
        }
        Ok(())
    }

    fn setup_vmcs(&mut self, entry: GuestPhysAddr, ept_root: HostPhysAddr) -> HyperResult {
        let paddr = self.vmcs.phys_addr() as u64;
        unsafe {
            vmx::vmclear(paddr)?;
        }
        self.bind_to_current_processor()?;
        self.setup_vmcs_host()?;
        self.setup_vmcs_guest(entry)?;
        self.setup_vmcs_control(ept_root)?;
        self.unbind_from_current_processor()?;
        Ok(())
    }

    fn setup_vmcs_host(&mut self) -> HyperResult {
        VmcsHost64::IA32_PAT.write(Msr::IA32_PAT.read())?;
        VmcsHost64::IA32_EFER.write(Msr::IA32_EFER.read())?;

        VmcsHostNW::CR0.write(Cr0::read_raw() as _)?;
        VmcsHostNW::CR3.write(Cr3::read_raw().0.start_address().as_u64() as _)?;
        VmcsHostNW::CR4.write(Cr4::read_raw() as _)?;

        VmcsHost16::ES_SELECTOR.write(x86::segmentation::es().bits())?;
        VmcsHost16::CS_SELECTOR.write(x86::segmentation::cs().bits())?;
        VmcsHost16::SS_SELECTOR.write(x86::segmentation::ss().bits())?;
        VmcsHost16::DS_SELECTOR.write(x86::segmentation::ds().bits())?;
        VmcsHost16::FS_SELECTOR.write(x86::segmentation::fs().bits())?;
        VmcsHost16::GS_SELECTOR.write(x86::segmentation::gs().bits())?;
        VmcsHostNW::FS_BASE.write(Msr::IA32_FS_BASE.read() as _)?;
        VmcsHostNW::GS_BASE.write(Msr::IA32_GS_BASE.read() as _)?;

        let tr = unsafe { x86::task::tr() };
        let mut gdtp = DescriptorTablePointer::<u64>::default();
        let mut idtp = DescriptorTablePointer::<u64>::default();
        unsafe {
            dtables::sgdt(&mut gdtp);
            dtables::sidt(&mut idtp);
        }
        VmcsHost16::TR_SELECTOR.write(tr.bits())?;
        VmcsHostNW::TR_BASE.write(get_tr_base(tr, &gdtp) as _)?;
        VmcsHostNW::GDTR_BASE.write(gdtp.base as _)?;
        VmcsHostNW::IDTR_BASE.write(idtp.base as _)?;
        VmcsHostNW::RIP.write(Self::vmx_exit as usize)?;

        VmcsHostNW::IA32_SYSENTER_ESP.write(0)?;
        VmcsHostNW::IA32_SYSENTER_EIP.write(0)?;
        VmcsHost32::IA32_SYSENTER_CS.write(0)?;
        Ok(())
    }

    fn setup_vmcs_guest(&mut self, entry: GuestPhysAddr) -> HyperResult {
        let cr0_guest = Cr0Flags::EXTENSION_TYPE | Cr0Flags::NUMERIC_ERROR;
        let cr0_host_owned = Cr0Flags::NUMERIC_ERROR; // | Cr0Flags::NOT_WRITE_THROUGH | Cr0Flags::CACHE_DISABLE;
        let cr0_read_shadow = Cr0Flags::NUMERIC_ERROR;
        VmcsGuestNW::CR0.write(cr0_guest.bits() as _)?;
        VmcsControlNW::CR0_GUEST_HOST_MASK.write(cr0_host_owned.bits() as _)?;
        VmcsControlNW::CR0_READ_SHADOW.write(cr0_read_shadow.bits() as _)?;

        let cr4_guest = Cr4Flags::VIRTUAL_MACHINE_EXTENSIONS;
        let cr4_host_owned = Cr4Flags::VIRTUAL_MACHINE_EXTENSIONS;
        let cr4_read_shadow = 0;
        VmcsGuestNW::CR4.write(cr4_guest.bits() as _)?;
        VmcsControlNW::CR4_GUEST_HOST_MASK.write(cr4_host_owned.bits() as _)?;
        VmcsControlNW::CR4_READ_SHADOW.write(cr4_read_shadow)?;

        macro_rules! set_guest_segment {
            ($seg: ident, $access_rights: expr) => {{
                use VmcsGuest16::*;
                use VmcsGuest32::*;
                use VmcsGuestNW::*;
                concat_idents!($seg, _SELECTOR).write(0)?;
                concat_idents!($seg, _BASE).write(0)?;
                concat_idents!($seg, _LIMIT).write(0xffff)?;
                concat_idents!($seg, _ACCESS_RIGHTS).write($access_rights)?;
            }};
        }

        set_guest_segment!(ES, 0x93); // 16-bit, present, data, read/write, accessed
        set_guest_segment!(CS, 0x9b); // 16-bit, present, code, exec/read, accessed
        set_guest_segment!(SS, 0x93);
        set_guest_segment!(DS, 0x93);
        set_guest_segment!(FS, 0x93);
        set_guest_segment!(GS, 0x93);
        set_guest_segment!(TR, 0x8b); // present, system, 32-bit TSS busy
        set_guest_segment!(LDTR, 0x82); // present, system, LDT

        VmcsGuestNW::GDTR_BASE.write(0)?;
        VmcsGuest32::GDTR_LIMIT.write(0xffff)?;
        VmcsGuestNW::IDTR_BASE.write(0)?;
        VmcsGuest32::IDTR_LIMIT.write(0xffff)?;

        VmcsGuestNW::CR3.write(0)?;
        VmcsGuestNW::DR7.write(0x400)?;
        VmcsGuestNW::RSP.write(0)?;
        VmcsGuestNW::RIP.write(entry)?;
        VmcsGuestNW::RFLAGS.write(0x2)?;
        VmcsGuestNW::PENDING_DBG_EXCEPTIONS.write(0)?;
        VmcsGuestNW::IA32_SYSENTER_ESP.write(0)?;
        VmcsGuestNW::IA32_SYSENTER_EIP.write(0)?;
        VmcsGuest32::IA32_SYSENTER_CS.write(0)?;

        VmcsGuest32::INTERRUPTIBILITY_STATE.write(0)?;
        VmcsGuest32::ACTIVITY_STATE.write(0)?;
        VmcsGuest32::VMX_PREEMPTION_TIMER_VALUE.write(0)?;

        VmcsGuest64::LINK_PTR.write(u64::MAX)?; // SDM Vol. 3C, Section 24.4.2
        VmcsGuest64::IA32_DEBUGCTL.write(0)?;
        VmcsGuest64::IA32_PAT.write(Msr::IA32_PAT.read())?;
        VmcsGuest64::IA32_EFER.write(0)?;
        Ok(())
    }

    fn setup_vmcs_control(&mut self, ept_root: HostPhysAddr) -> HyperResult {
        // Intercept NMI and external interrupts.
        use super::vmcs::controls::*;
        use PinbasedControls as PinCtrl;
        vmcs::set_control(
            VmcsControl32::PINBASED_EXEC_CONTROLS,
            Msr::IA32_VMX_TRUE_PINBASED_CTLS,
            Msr::IA32_VMX_PINBASED_CTLS.read() as u32,
            (PinCtrl::NMI_EXITING | PinCtrl::EXTERNAL_INTERRUPT_EXITING).bits(),
            0,
        )?;

        // Intercept all I/O instructions, use MSR bitmaps, activate secondary controls,
        // disable CR3 load/store interception.
        use PrimaryControls as CpuCtrl;
        vmcs::set_control(
            VmcsControl32::PRIMARY_PROCBASED_EXEC_CONTROLS,
            Msr::IA32_VMX_TRUE_PROCBASED_CTLS,
            Msr::IA32_VMX_PROCBASED_CTLS.read() as u32,
            (CpuCtrl::UNCOND_IO_EXITING | CpuCtrl::USE_MSR_BITMAPS | CpuCtrl::SECONDARY_CONTROLS)
                .bits(),
            (CpuCtrl::CR3_LOAD_EXITING
                | CpuCtrl::CR3_STORE_EXITING
                | CpuCtrl::CR8_LOAD_EXITING
                | CpuCtrl::CR8_STORE_EXITING)
                .bits(),
        )?;

        // Enable EPT, RDTSCP, INVPCID, and unrestricted guest.
        use SecondaryControls as CpuCtrl2;
        vmcs::set_control(
            VmcsControl32::SECONDARY_PROCBASED_EXEC_CONTROLS,
            Msr::IA32_VMX_PROCBASED_CTLS2,
            0,
            (CpuCtrl2::ENABLE_EPT
                | CpuCtrl2::ENABLE_RDTSCP
                | CpuCtrl2::ENABLE_INVPCID
                | CpuCtrl2::UNRESTRICTED_GUEST
                | CpuCtrl2::ENABLE_XSAVES_XRSTORS)
                .bits(),
            0,
        )?;

        // Switch to 64-bit host, acknowledge interrupt info, switch IA32_PAT/IA32_EFER on VM exit.
        use ExitControls as ExitCtrl;
        vmcs::set_control(
            VmcsControl32::VMEXIT_CONTROLS,
            Msr::IA32_VMX_TRUE_EXIT_CTLS,
            Msr::IA32_VMX_EXIT_CTLS.read() as u32,
            (ExitCtrl::HOST_ADDRESS_SPACE_SIZE
                | ExitCtrl::ACK_INTERRUPT_ON_EXIT
                | ExitCtrl::SAVE_IA32_PAT
                | ExitCtrl::LOAD_IA32_PAT
                | ExitCtrl::SAVE_IA32_EFER
                | ExitCtrl::LOAD_IA32_EFER)
                .bits(),
            0,
        )?;

        // Load guest IA32_PAT/IA32_EFER on VM entry.
        use EntryControls as EntryCtrl;
        vmcs::set_control(
            VmcsControl32::VMENTRY_CONTROLS,
            Msr::IA32_VMX_TRUE_ENTRY_CTLS,
            Msr::IA32_VMX_ENTRY_CTLS.read() as u32,
            (EntryCtrl::LOAD_IA32_PAT | EntryCtrl::LOAD_IA32_EFER).bits(),
            0,
        )?;

        vmcs::set_ept_pointer(ept_root)?;

        // No MSR switches if hypervisor doesn't use and there is only one vCPU.
        VmcsControl32::VMEXIT_MSR_STORE_COUNT.write(0)?;
        VmcsControl32::VMEXIT_MSR_LOAD_COUNT.write(0)?;
        VmcsControl32::VMENTRY_MSR_LOAD_COUNT.write(0)?;

        // Pass-through exceptions (except #UD(6)), don't use I/O bitmap, set MSR bitmaps.
        let exception_bitmap: u32 = 1 << 6;

        VmcsControl32::EXCEPTION_BITMAP.write(exception_bitmap)?;
        VmcsControl64::IO_BITMAP_A_ADDR.write(0)?;
        VmcsControl64::IO_BITMAP_B_ADDR.write(0)?;
        VmcsControl64::MSR_BITMAPS_ADDR.write(self.msr_bitmap.phys_addr() as _)?;
        Ok(())
    }

    fn get_paging_level(&self) -> usize {
        let mut level: u32 = 0; // non-paging
        let cr0 = VmcsGuestNW::CR0.read().unwrap();
        let cr4 = VmcsGuestNW::CR4.read().unwrap();
        let efer = VmcsGuest64::IA32_EFER.read().unwrap();
        // paging is enabled
        if cr0 & Cr0Flags::PAGING.bits() as usize != 0 {
            if cr4 & Cr4Flags::PHYSICAL_ADDRESS_EXTENSION.bits() as usize != 0 {
                // is long mode
                if efer & EferFlags::LONG_MODE_ACTIVE.bits() != 0 {
                    level = 4;
                } else {
                    level = 3;
                }
            } else {
                level = 2;
            }
        }
        level as usize
    }
}

// Implementaton for type1.5 hypervisor
#[cfg(feature = "type1_5")]
impl<H: HyperCraftHal> VmxVcpu<H> {
    /// Create a new [`VmxVcpu`] for type1.5 hypervisor
    pub fn new_type15(
        vcpu_id: usize,
        vmcs_revision_id: u32,
        ept_root: HostPhysAddr,
        linux: &LinuxContext,
    ) -> HyperResult<Self> {
        XState::enable_xsave();
        // unsafe { Cr4::write(Cr4::read() & !(Cr4Flags::OSXSAVE)) };
        let mut vcpu = Self {
            guest_regs: GeneralRegisters::default(),
            host_stack_top: 0,
            vcpu_id,
            launched: false,
            vmcs: VmxRegion::new(vmcs_revision_id, false)?,
            msr_bitmap: MsrBitmap::passthrough_all()?,
            pending_events: VecDeque::with_capacity(8),
            xstate: XState::new(),
        };
        // vcpu.setup_type15_msr_bitmap()?;
        vcpu.setup_type15_vmcs(ept_root, linux)?;
        let regs = vcpu.regs_mut();
        regs.rax = 0;
        regs.rbx = linux.rbx;
        regs.rbp = linux.rbp;
        regs.r12 = linux.r12;
        regs.r13 = linux.r13;
        regs.r14 = linux.r14;
        regs.r15 = linux.r15;

        info!("[HV] created VmxVcpu(vmcs: {:#x})", vcpu.vmcs.phys_addr());
        Ok(vcpu)
    }

    /// Run the guest. It returns when a vm-exit happens and returns the vm-exit if it cannot be handled by this [`VmxVcpu`] itself.
    pub fn run_type15(&mut self, _linux: &LinuxContext) -> Option<VmxExitInfo> {
        // Inject pending events
        if self.launched {
            self.inject_pending_events().unwrap();
        }

        // Run guest
        self.load_guest_xstate();
        // debug!("vcpu set to linux regs: {:#x?}", self.guest_regs);
        unsafe {
            if self.launched {
                // debug!("before resume");
                self.vmx_resume();
            } else {
                self.launched = true;
                VmcsHostNW::RSP
                    .write(&self.host_stack_top as *const _ as usize)
                    .unwrap();
                debug!("vcpu{} before vmlaunch", self.vcpu_id);
                self.vmx_launch();
            }
        }
        self.load_host_xstate();

        // Handle vm-exits
        let exit_info = self.exit_info().unwrap();
        trace!("VM exit: {:#x?}", exit_info);
        // debug!("VM exit: {:#x?}", exit_info);

        let cr4 = VmcsGuestNW::CR4.read().unwrap();
        if cr4.get_bit(18) {
            // debug!("get cr4 osxsave bit");
            // panic!("osxsave dead!");
        }
        match self.builtin_vmexit_handler(&exit_info) {
            Some(result) => {
                if result.is_err() {
                    panic!("VmxVcpu failed to handle a VM-exit that should be handled by itself: {:?}, error {:?}, vcpu: {:#x?}", exit_info.exit_reason, result.unwrap_err(), self);
                }

                None
            }
            None => Some(exit_info),
        }
    }

    fn setup_type15_msr_bitmap(&mut self) -> HyperResult {
        // read
        self.msr_bitmap.set_read_intercept(0x277, true); // IA32_PAT
        self.msr_bitmap.set_read_intercept(0x2FF, true); // IA32_MTRR_DEF_TYPE

        self.msr_bitmap.set_read_intercept(0x802, true); // IA32_X2APIC_APICID
        self.msr_bitmap.set_read_intercept(0x803, true); // IA32_X2APIC_VERSION
        self.msr_bitmap.set_read_intercept(0x808, true); // IA32_X2APIC_TPR
        self.msr_bitmap.set_read_intercept(0x80A, true); // IA32_X2APIC_PPR
        self.msr_bitmap.set_read_intercept(0x80D, true); // IA32_X2APIC_LDR
        self.msr_bitmap.set_read_intercept(0x80F, true); // IA32_X2APIC_SIVR
                                                         // IA32_X2APIC_ISR0..IA32_X2APIC_ISR7
        for msr in 0x810..=0x817 {
            self.msr_bitmap.set_read_intercept(msr, true);
        }
        // IA32_X2APIC_TMR0..IA32_X2APIC_TMR7
        for msr in 0x818..=0x81F {
            self.msr_bitmap.set_read_intercept(msr, true);
        }
        // IA32_X2APIC_IRR0..IA32_X2APIC_IRR7
        for msr in 0x820..=0x827 {
            self.msr_bitmap.set_read_intercept(msr, true);
        }
        self.msr_bitmap.set_read_intercept(0x828, true); // IA32_X2APIC_ESR
        self.msr_bitmap.set_read_intercept(0x82F, true); // IA32_X2APIC_LVT_CMCI
        self.msr_bitmap.set_read_intercept(0x830, true); // IA32_X2APIC_ICR
                                                         // IA32_X2APIC_LVT_*
        for msr in 0x832..=0x837 {
            self.msr_bitmap.set_read_intercept(msr, true);
        }
        self.msr_bitmap.set_read_intercept(0x838, true); // IA32_X2APIC_INIT_COUNT
        self.msr_bitmap.set_read_intercept(0x839, true); // IA32_X2APIC_CUR_COUNT
        self.msr_bitmap.set_read_intercept(0x83E, true); // IA32_X2APIC_DIV_CONF

        // write
        self.msr_bitmap.set_write_intercept(0x1B, true); // IA32_APIC_BASE

        // IA32_MTRR_*
        for msr in 0x200..=0x277 {
            self.msr_bitmap.set_write_intercept(msr, true);
        }
        self.msr_bitmap.set_write_intercept(0x277, true); // IA32_PAT
        self.msr_bitmap.set_write_intercept(0x2FF, true); // IA32_MTRR_DEF_TYPE
        self.msr_bitmap.set_write_intercept(0x38F, true); // IA32_PERF_GLOBAL_CTRL
        for msr in 0xC80..=0xD8F {
            self.msr_bitmap.set_write_intercept(msr, true);
        }

        self.msr_bitmap.set_write_intercept(0x808, true); // IA32_X2APIC_TPR
        self.msr_bitmap.set_write_intercept(0x80B, true); // IA32_X2APIC_EOI
        self.msr_bitmap.set_write_intercept(0x80F, true); // IA32_X2APIC_SIVR
        self.msr_bitmap.set_write_intercept(0x828, true); // IA32_X2APIC_ESR
        self.msr_bitmap.set_write_intercept(0x82F, true); // IA32_X2APIC_LVT_CMCI
        self.msr_bitmap.set_write_intercept(0x830, true); // IA32_X2APIC_ICR
                                                          // IA32_X2APIC_LVT_*
        for msr in 0x832..=0x837 {
            self.msr_bitmap.set_write_intercept(msr, true);
        }
        self.msr_bitmap.set_write_intercept(0x838, true); // IA32_X2APIC_INIT_COUNT
        self.msr_bitmap.set_write_intercept(0x839, true); // IA32_X2APIC_CUR_COUNT
        self.msr_bitmap.set_write_intercept(0x83E, true); // IA32_X2APIC_DIV_CONF

        Ok(())
    }

    fn setup_type15_vmcs(&mut self, ept_root: HostPhysAddr, linux: &LinuxContext) -> HyperResult {
        let paddr = self.vmcs.phys_addr() as u64;
        unsafe {
            vmx::vmclear(paddr)?;
        }
        self.bind_to_current_processor()?;
        self.setup_type15_vmcs_host()?;
        self.setup_type15_vmcs_guest(linux)?;
        self.setup_type15_vmcs_control(ept_root)?;
        self.unbind_from_current_processor()?;
        Ok(())
    }

    fn setup_type15_vmcs_host(&mut self) -> HyperResult {
        use x86::Ring;
        VmcsHost64::IA32_PAT.write(Msr::IA32_PAT.read())?;
        VmcsHost64::IA32_EFER.write(Msr::IA32_EFER.read())?;

        VmcsHostNW::CR0.write(Cr0::read_raw() as _)?;
        VmcsHostNW::CR3.write(Cr3::read_raw().0.start_address().as_u64() as _)?;
        VmcsHostNW::CR4.write(Cr4::read_raw() as _)?;

        VmcsHost16::ES_SELECTOR.write(0)?;
        VmcsHost16::CS_SELECTOR.write(SegmentSelector::new(2, Ring::Ring0).bits())?;
        VmcsHost16::SS_SELECTOR.write(0)?;
        VmcsHost16::DS_SELECTOR.write(0)?;
        VmcsHost16::FS_SELECTOR.write(0)?;
        VmcsHost16::GS_SELECTOR.write(0)?;
        VmcsHost16::TR_SELECTOR.write(SegmentSelector::new(7, Ring::Ring0).bits())?;
        VmcsHostNW::FS_BASE.write(0)?;
        VmcsHostNW::GS_BASE.write(Msr::IA32_GS_BASE.read() as _)?;
        VmcsHostNW::TR_BASE.write(0)?;

        let mut gdtp = DescriptorTablePointer::<u64>::default();
        let mut idtp = DescriptorTablePointer::<u64>::default();
        unsafe {
            dtables::sgdt(&mut gdtp);
            dtables::sidt(&mut idtp);
        }
        VmcsHostNW::GDTR_BASE.write(gdtp.base as _)?;
        VmcsHostNW::IDTR_BASE.write(idtp.base as _)?;

        VmcsHostNW::IA32_SYSENTER_ESP.write(0)?;
        VmcsHostNW::IA32_SYSENTER_EIP.write(0)?;
        VmcsHost32::IA32_SYSENTER_CS.write(0)?;

        // TODO: RSP
        VmcsHostNW::RIP.write(Self::vmx_exit as usize)?;

        Ok(())
    }

    fn setup_type15_vmcs_guest(&mut self, linux: &LinuxContext) -> HyperResult {
        VmcsGuest64::IA32_PAT.write(linux.pat)?;
        VmcsGuest64::IA32_EFER.write(linux.efer)?;

        self.set_cr(0, linux.cr0.bits());
        self.set_cr(4, linux.cr4.bits());
        self.set_cr(3, linux.cr3);

        macro_rules! set_guest_segment {
            ($seg: expr, $reg: ident) => {{
                use VmcsGuest16::*;
                use VmcsGuest32::*;
                use VmcsGuestNW::*;
                concat_idents!($reg, _SELECTOR).write($seg.selector.bits())?;
                concat_idents!($reg, _BASE).write($seg.base as _)?;
                concat_idents!($reg, _LIMIT).write($seg.limit)?;
                concat_idents!($reg, _ACCESS_RIGHTS).write($seg.access_rights.bits())?;
            }};
        }
        set_guest_segment!(linux.es, ES);
        set_guest_segment!(linux.cs, CS);
        set_guest_segment!(linux.ss, SS);
        set_guest_segment!(linux.ds, DS);
        set_guest_segment!(linux.fs, FS);
        set_guest_segment!(linux.gs, GS);
        set_guest_segment!(linux.tss, TR);
        set_guest_segment!(Segment::invalid(), LDTR);

        VmcsGuestNW::GDTR_BASE.write(linux.gdt.base as _)?;
        VmcsGuest32::GDTR_LIMIT.write(linux.gdt.limit as _)?;
        VmcsGuestNW::IDTR_BASE.write(linux.idt.base as _)?;
        VmcsGuest32::IDTR_LIMIT.write(linux.idt.limit as _)?;

        debug!(
            "this is the linux rip: {:#x} rsp:{:#x}",
            linux.rip, linux.rsp
        );
        VmcsGuestNW::RSP.write(linux.rsp as _)?;
        VmcsGuestNW::RIP.write(linux.rip as _)?;
        VmcsGuestNW::RFLAGS.write(0x2)?;

        VmcsGuest32::IA32_SYSENTER_CS.write(Msr::IA32_SYSENTER_CS.read() as _)?;
        VmcsGuestNW::IA32_SYSENTER_ESP.write(Msr::IA32_SYSENTER_ESP.read() as _)?;
        VmcsGuestNW::IA32_SYSENTER_EIP.write(Msr::IA32_SYSENTER_EIP.read() as _)?;

        VmcsGuestNW::DR7.write(0x400)?;
        VmcsGuest64::IA32_DEBUGCTL.write(0)?;

        VmcsGuest32::ACTIVITY_STATE.write(0)?;
        VmcsGuest32::INTERRUPTIBILITY_STATE.write(0)?;
        VmcsGuestNW::PENDING_DBG_EXCEPTIONS.write(0)?;

        VmcsGuest64::LINK_PTR.write(u64::MAX)?;
        VmcsGuest32::VMX_PREEMPTION_TIMER_VALUE.write(0)?;
        Ok(())
    }

    fn setup_type15_vmcs_control(&mut self, ept_root: HostPhysAddr) -> HyperResult {
        use super::vmcs::controls::*;
        use PinbasedControls as PinCtrl;
        vmcs::set_control_type15(
            VmcsControl32::PINBASED_EXEC_CONTROLS,
            Msr::IA32_VMX_PINBASED_CTLS.read(),
            // NO INTR_EXITING to pass-through interrupts
            PinCtrl::NMI_EXITING.bits(),
            0,
        )?;

        // Intercept all I/O instructions, use MSR bitmaps, activate secondary controls,
        // disable CR3 load/store interception.
        use PrimaryControls as CpuCtrl;
        vmcs::set_control_type15(
            VmcsControl32::PRIMARY_PROCBASED_EXEC_CONTROLS,
            Msr::IA32_VMX_PROCBASED_CTLS.read(),
            // NO UNCOND_IO_EXITING to pass-through PIO
            (CpuCtrl::USE_MSR_BITMAPS | CpuCtrl::SECONDARY_CONTROLS).bits(),
            (CpuCtrl::CR3_LOAD_EXITING | CpuCtrl::CR3_STORE_EXITING).bits(),
        )?;

        // Enable EPT, RDTSCP, INVPCID, and unrestricted guest.
        use SecondaryControls as CpuCtrl2;
        let mut val = CpuCtrl2::ENABLE_EPT | CpuCtrl2::UNRESTRICTED_GUEST;
        if let Some(features) = CpuId::new().get_extended_processor_and_feature_identifiers() {
            if features.has_rdtscp() {
                val |= CpuCtrl2::ENABLE_RDTSCP;
            }
        }
        if let Some(features) = CpuId::new().get_extended_feature_info() {
            if features.has_invpcid() {
                val |= CpuCtrl2::ENABLE_INVPCID;
            }
        }
        if let Some(features) = CpuId::new().get_extended_state_info() {
            if features.has_xsaves_xrstors() {
                val |= CpuCtrl2::ENABLE_XSAVES_XRSTORS;
            }
        }
        vmcs::set_control_type15(
            VmcsControl32::SECONDARY_PROCBASED_EXEC_CONTROLS,
            Msr::IA32_VMX_PROCBASED_CTLS2.read(),
            val.bits(),
            0,
        )?;

        // Switch to 64-bit host, acknowledge interrupt info, switch IA32_PAT/IA32_EFER on VM exit.
        use ExitControls as ExitCtrl;
        vmcs::set_control_type15(
            VmcsControl32::VMEXIT_CONTROLS,
            Msr::IA32_VMX_EXIT_CTLS.read(),
            (ExitCtrl::HOST_ADDRESS_SPACE_SIZE
                | ExitCtrl::SAVE_IA32_PAT
                | ExitCtrl::LOAD_IA32_PAT
                | ExitCtrl::SAVE_IA32_EFER
                | ExitCtrl::LOAD_IA32_EFER)
                .bits(),
            0,
        )?;

        // Load guest IA32_PAT/IA32_EFER on VM entry.
        use EntryControls as EntryCtrl;
        vmcs::set_control_type15(
            VmcsControl32::VMENTRY_CONTROLS,
            Msr::IA32_VMX_ENTRY_CTLS.read(),
            (EntryCtrl::IA32E_MODE_GUEST | EntryCtrl::LOAD_IA32_PAT | EntryCtrl::LOAD_IA32_EFER)
                .bits(),
            0,
        )?;

        // No MSR switches if hypervisor doesn't use and there is only one vCPU.
        VmcsControl32::VMEXIT_MSR_STORE_COUNT.write(0)?;
        VmcsControl32::VMEXIT_MSR_LOAD_COUNT.write(0)?;
        VmcsControl32::VMENTRY_MSR_LOAD_COUNT.write(0)?;

        VmcsControlNW::CR4_GUEST_HOST_MASK.write(0)?;
        VmcsControl32::CR3_TARGET_COUNT.write(0)?;

        vmcs::set_ept_pointer(ept_root)?;

        // Pass-through exceptions (except #UD(6)), don't use I/O bitmap, set MSR bitmaps.
        let exception_bitmap: u32 = 1 << 6;

        VmcsControl32::EXCEPTION_BITMAP.write(exception_bitmap)?;
        VmcsControl64::MSR_BITMAPS_ADDR.write(self.msr_bitmap.phys_addr() as _)?;
        Ok(())
    }

    fn set_cr(&mut self, cr_idx: usize, val: u64) {
        (|| -> HyperResult {
            match cr_idx {
                0 => {
                    // Retrieve/validate restrictions on CR0
                    //
                    // In addition to what the VMX MSRs tell us, make sure that
                    // - NW and CD are kept off as they are not updated on VM exit and we
                    //   don't want them enabled for performance reasons while in root mode
                    // - PE and PG can be freely chosen (by the guest) because we demand
                    //   unrestricted guest mode support anyway
                    // - ET is ignored
                    let must0 = Msr::IA32_VMX_CR0_FIXED1.read()
                        & !(Cr0Flags::NOT_WRITE_THROUGH | Cr0Flags::CACHE_DISABLE).bits();
                    let must1 = Msr::IA32_VMX_CR0_FIXED0.read()
                        & !(Cr0Flags::PAGING | Cr0Flags::PROTECTED_MODE_ENABLE).bits();
                    VmcsGuestNW::CR0.write(((val & must0) | must1) as _)?;
                    VmcsControlNW::CR0_READ_SHADOW.write(val as _)?;
                    VmcsControlNW::CR0_GUEST_HOST_MASK.write((must1 | !must0) as _)?;
                }
                3 => VmcsGuestNW::CR3.write(val as _)?,
                4 => {
                    // Retrieve/validate restrictions on CR4
                    let must0 = Msr::IA32_VMX_CR4_FIXED1.read();
                    let must1 = Msr::IA32_VMX_CR4_FIXED0.read();
                    let val = val | Cr4Flags::VIRTUAL_MACHINE_EXTENSIONS.bits();
                    VmcsGuestNW::CR4.write(((val & must0) | must1) as _)?;
                    VmcsControlNW::CR4_READ_SHADOW.write(val as _)?;
                    VmcsControlNW::CR4_GUEST_HOST_MASK.write((must1 | !must0) as _)?;
                }
                _ => unreachable!(),
            };
            Ok(())
        })()
        .expect("Failed to write guest control register")
    }

    fn cr(&self, cr_idx: usize) -> usize {
        (|| -> HyperResult<usize> {
            Ok(match cr_idx {
                0 => VmcsGuestNW::CR0.read()?,
                3 => VmcsGuestNW::CR3.read()?,
                4 => {
                    let host_mask = VmcsControlNW::CR4_GUEST_HOST_MASK.read()?;
                    (VmcsControlNW::CR4_READ_SHADOW.read()? & host_mask)
                        | (VmcsGuestNW::CR4.read()? & !host_mask)
                }
                _ => unreachable!(),
            })
        })()
        .expect("Failed to read guest control register")
    }

    /// Create a new [`VmxVcpu`].
    pub fn new_nimbos(
        vcpu_id: usize,
        vmcs_revision_id: u32,
        entry: GuestPhysAddr,
        ept_root: HostPhysAddr,
    ) -> HyperResult<Self> {
        XState::enable_xsave();
        let mut vcpu = Self {
            guest_regs: GeneralRegisters::default(),
            host_stack_top: 0,
            vcpu_id,
            launched: false,
            vmcs: VmxRegion::new(vmcs_revision_id, false)?,
            msr_bitmap: MsrBitmap::passthrough_all()?,
            pending_events: VecDeque::with_capacity(8),
            xstate: XState::new(),
        };
        // vcpu.setup_msr_bitmap()?;
        vcpu.setup_nimbos_vmcs(entry, ept_root)?;
        info!("[HV] created VmxVcpu(vmcs: {:#x})", vcpu.vmcs.phys_addr());
        Ok(vcpu)
    }

    fn setup_nimbos_vmcs(&mut self, entry: GuestPhysAddr, ept_root: HostPhysAddr) -> HyperResult {
        let paddr = self.vmcs.phys_addr() as u64;
        unsafe {
            vmx::vmclear(paddr)?;
        }
        self.bind_to_current_processor()?;
        self.setup_vmcs_host()?;
        self.setup_vmcs_guest(entry)?;
        self.setup_nimbos_vmcs_control(ept_root)?;
        self.unbind_from_current_processor()?;
        Ok(())
    }
    fn setup_nimbos_vmcs_control(&mut self, ept_root: HostPhysAddr) -> HyperResult {
        // Intercept NMI and external interrupts.
        use super::vmcs::controls::*;
        use PinbasedControls as PinCtrl;
        vmcs::set_control(
            VmcsControl32::PINBASED_EXEC_CONTROLS,
            Msr::IA32_VMX_TRUE_PINBASED_CTLS,
            Msr::IA32_VMX_PINBASED_CTLS.read() as u32,
            (PinCtrl::NMI_EXITING).bits(),
            0,
        )?;

        // Intercept all I/O instructions, use MSR bitmaps, activate secondary controls,
        // disable CR3 load/store interception.
        use PrimaryControls as CpuCtrl;
        vmcs::set_control(
            VmcsControl32::PRIMARY_PROCBASED_EXEC_CONTROLS,
            Msr::IA32_VMX_TRUE_PROCBASED_CTLS,
            Msr::IA32_VMX_PROCBASED_CTLS.read() as u32,
            (CpuCtrl::UNCOND_IO_EXITING | CpuCtrl::USE_MSR_BITMAPS | CpuCtrl::SECONDARY_CONTROLS)
                .bits(),
            (CpuCtrl::CR3_LOAD_EXITING
                | CpuCtrl::CR3_STORE_EXITING
                | CpuCtrl::CR8_LOAD_EXITING
                | CpuCtrl::CR8_STORE_EXITING)
                .bits(),
        )?;

        // Enable EPT, RDTSCP, INVPCID, and unrestricted guest.
        use SecondaryControls as CpuCtrl2;
        vmcs::set_control(
            VmcsControl32::SECONDARY_PROCBASED_EXEC_CONTROLS,
            Msr::IA32_VMX_PROCBASED_CTLS2,
            0,
            (CpuCtrl2::ENABLE_EPT
                | CpuCtrl2::ENABLE_RDTSCP
                | CpuCtrl2::ENABLE_INVPCID
                | CpuCtrl2::UNRESTRICTED_GUEST
                | CpuCtrl2::ENABLE_XSAVES_XRSTORS)
                .bits(),
            0,
        )?;

        // Switch to 64-bit host, acknowledge interrupt info, switch IA32_PAT/IA32_EFER on VM exit.
        use ExitControls as ExitCtrl;
        vmcs::set_control(
            VmcsControl32::VMEXIT_CONTROLS,
            Msr::IA32_VMX_TRUE_EXIT_CTLS,
            Msr::IA32_VMX_EXIT_CTLS.read() as u32,
            (ExitCtrl::HOST_ADDRESS_SPACE_SIZE
                | ExitCtrl::ACK_INTERRUPT_ON_EXIT
                | ExitCtrl::SAVE_IA32_PAT
                | ExitCtrl::LOAD_IA32_PAT
                | ExitCtrl::SAVE_IA32_EFER
                | ExitCtrl::LOAD_IA32_EFER)
                .bits(),
            0,
        )?;

        // Load guest IA32_PAT/IA32_EFER on VM entry.
        use EntryControls as EntryCtrl;
        vmcs::set_control(
            VmcsControl32::VMENTRY_CONTROLS,
            Msr::IA32_VMX_TRUE_ENTRY_CTLS,
            Msr::IA32_VMX_ENTRY_CTLS.read() as u32,
            (EntryCtrl::LOAD_IA32_PAT | EntryCtrl::LOAD_IA32_EFER).bits(),
            0,
        )?;

        vmcs::set_ept_pointer(ept_root)?;

        // No MSR switches if hypervisor doesn't use and there is only one vCPU.
        VmcsControl32::VMEXIT_MSR_STORE_COUNT.write(0)?;
        VmcsControl32::VMEXIT_MSR_LOAD_COUNT.write(0)?;
        VmcsControl32::VMENTRY_MSR_LOAD_COUNT.write(0)?;

        // Pass-through exceptions (except #UD(6)), don't use I/O bitmap, set MSR bitmaps.
        let exception_bitmap: u32 = 1 << 6;

        VmcsControl32::EXCEPTION_BITMAP.write(exception_bitmap)?;
        VmcsControl64::IO_BITMAP_A_ADDR.write(0)?;
        VmcsControl64::IO_BITMAP_B_ADDR.write(0)?;
        VmcsControl64::MSR_BITMAPS_ADDR.write(self.msr_bitmap.phys_addr() as _)?;
        Ok(())
    }
}

/// Get ready then vmlaunch or vmresume.
macro_rules! vmx_entry_with {
    ($instr:literal) => {
        asm!(
            save_regs_to_stack!(),                  // save host status
            "mov    [rdi + {host_stack_top}], rsp", // save current RSP to Vcpu::host_stack_top
            "mov    rsp, rdi",                      // set RSP to guest regs area
            restore_regs_from_stack!(),             // restore guest status
            $instr,                                 // let's go!
            "jmp    {failed}",
            host_stack_top = const size_of::<GeneralRegisters>(),
            failed = sym Self::vmx_entry_failed,
            options(noreturn),
        )
    }
}

impl<H: HyperCraftHal> VmxVcpu<H> {
    #[naked]
    /// Enter guest with vmlaunch.
    ///
    /// `#[naked]` is essential here, without it the rust compiler will think `&mut self` is not used and won't give us correct %rdi.
    ///
    /// This function itself never returns, but [`Self::vmx_exit`] will do the return for this.
    ///
    /// The return value is a dummy value.
    unsafe extern "C" fn vmx_launch(&mut self) -> usize {
        vmx_entry_with!("vmlaunch")
    }

    #[naked]
    /// Enter guest with vmresume.
    ///
    /// See [`Self::vmx_launch`] for detail.
    unsafe extern "C" fn vmx_resume(&mut self) -> usize {
        vmx_entry_with!("vmresume")
    }

    #[naked]
    /// Return after vm-exit.
    ///
    /// The return value is a dummy value.
    unsafe extern "C" fn vmx_exit(&mut self) -> usize {
        asm!(
            save_regs_to_stack!(),                  // save guest status
            "mov    rsp, [rsp + {host_stack_top}]", // set RSP to Vcpu::host_stack_top
            restore_regs_from_stack!(),             // restore host status
            "ret",
            host_stack_top = const size_of::<GeneralRegisters>(),
            options(noreturn),
        );
    }

    fn vmx_entry_failed() -> ! {
        panic!("{}", vmcs::instruction_error().as_str())
    }

    /// Whether the guest interrupts are blocked. (SDM Vol. 3C, Section 24.4.2, Table 24-3)
    fn allow_interrupt(&self) -> bool {
        let rflags = VmcsGuestNW::RFLAGS.read().unwrap();
        let block_state = VmcsGuest32::INTERRUPTIBILITY_STATE.read().unwrap();
        rflags as u64 & x86_64::registers::rflags::RFlags::INTERRUPT_FLAG.bits() != 0
            && block_state == 0
    }

    /// Try to inject a pending event before next VM entry.
    fn inject_pending_events(&mut self) -> HyperResult {
        if let Some(event) = self.pending_events.front() {
            if event.0 < 32 || self.allow_interrupt() {
                // if it's an exception, or an interrupt that is not blocked, inject it directly.
                vmcs::inject_event(event.0, event.1)?;
                self.pending_events.pop_front();
            } else {
                // interrupts are blocked, enable interrupt-window exiting.
                self.set_interrupt_window(true)?;
            }
        }
        Ok(())
    }

    /// Handle vm-exits than can and should be handled by [`VmxVcpu`] itself.
    ///
    /// Return the result or None if the vm-exit was not handled.
    fn builtin_vmexit_handler(&mut self, exit_info: &VmxExitInfo) -> Option<HyperResult> {
        if exit_info.entry_failure {
            panic!("VM entry failed: {:#x?}", exit_info);
        }

        // Following vm-exits are handled here:
        // - interrupt window: turn off interrupt window;
        // - xsetbv: set guest xcr;
        // - cr access: just panic;
        match exit_info.exit_reason {
            VmxExitReason::INTERRUPT_WINDOW => Some(self.set_interrupt_window(false)),
            VmxExitReason::XSETBV => Some(self.handle_xsetbv()),
            VmxExitReason::CR_ACCESS => panic!(
                "Guest's access to cr not allowed: {:#x?}, {:#x?}",
                self,
                vmcs::cr_access_info()
            ),
            VmxExitReason::CPUID => Some(self.handle_cpuid()),
            _ => None,
        }
    }

    #[cfg(feature = "type1_5")]
    fn handle_cpuid(&mut self) -> HyperResult {
        use raw_cpuid::{cpuid, CpuIdResult};

        const VM_EXIT_INSTR_LEN_CPUID: u8 = 2;
        const LEAF_FEATURE_INFO: u32 = 0x1;
        const LEAF_HYPERVISOR_INFO: u32 = 0x4000_0000;
        const LEAF_HYPERVISOR_FEATURE: u32 = 0x4000_0001;
        const VENDOR_STR: &[u8; 12] = b"RVMRVMRVMRVM";

        const OSXSAVE: u64 = 1 << 27;
        const VMX: u64 = 1 << 5;
        const HYPERVISOR: u64 = 1 << 31;

        let signature = unsafe { &*("RVMRVMRVMRVM".as_ptr() as *const [u32; 3]) };
        let cr4_flags = Cr4Flags::from_bits_truncate(self.cr(4) as u64);
        let guest_regs = self.regs_mut();
        let function = guest_regs.rax as u32;
        if function == LEAF_HYPERVISOR_INFO as _ {
            guest_regs.rax = LEAF_HYPERVISOR_FEATURE as u32 as _;
            guest_regs.rbx = signature[0] as _;
            guest_regs.rcx = signature[1] as _;
            guest_regs.rdx = signature[2] as _;
        } else if function == LEAF_HYPERVISOR_FEATURE as _ {
            guest_regs.rax = 0;
            guest_regs.rbx = 0;
            guest_regs.rcx = 0;
            guest_regs.rdx = 0;
        } else {
            let res = cpuid!(guest_regs.rax, guest_regs.rcx);
            guest_regs.rax = res.eax as _;
            guest_regs.rbx = res.ebx as _;
            guest_regs.rcx = res.ecx as _;
            guest_regs.rdx = res.edx as _;
            if function == LEAF_FEATURE_INFO as _ {
                let mut flags = guest_regs.rcx;
                if cr4_flags.contains(Cr4Flags::OSXSAVE) {
                    flags |= OSXSAVE;
                }
                flags |= !VMX;
                flags |= HYPERVISOR;
                guest_regs.rcx = flags;
            }
        }
        // debug!("cpuid return guest_regs:{:#x?}", guest_regs);
        self.advance_rip(VM_EXIT_INSTR_LEN_CPUID)?;
        Ok(())
    }

    #[cfg(not(feature = "type1_5"))]
    fn handle_cpuid(&mut self) -> HyperResult {
        use raw_cpuid::{cpuid, CpuIdResult};

        const VM_EXIT_INSTR_LEN_CPUID: u8 = 2;
        const LEAF_FEATURE_INFO: u32 = 0x1;
        const LEAF_STRUCTURED_EXTENDED_FEATURE_FLAGS_ENUMERATION: u32 = 0x7;
        const LEAF_PROCESSOR_EXTENDED_STATE_ENUMERATION: u32 = 0xd;
        const LEAF_HYPERVISOR_INFO: u32 = 0x4000_0000;
        const LEAF_HYPERVISOR_FEATURE: u32 = 0x4000_0001;
        const VENDOR_STR: &[u8; 12] = b"RVMRVMRVMRVM";
        let vendor_regs = unsafe { &*(VENDOR_STR.as_ptr() as *const [u32; 3]) };

        let regs_clone = self.regs_mut().clone();
        let function = regs_clone.rax as u32;
        let res = match function {
            LEAF_FEATURE_INFO => {
                const FEATURE_VMX: u32 = 1 << 5;
                const FEATURE_HYPERVISOR: u32 = 1 << 31;
                let mut res = cpuid!(regs_clone.rax, regs_clone.rcx);
                res.ecx &= !FEATURE_VMX;
                res.ecx |= FEATURE_HYPERVISOR;
                res
            }
            LEAF_STRUCTURED_EXTENDED_FEATURE_FLAGS_ENUMERATION => {
                let mut res = cpuid!(regs_clone.rax, regs_clone.rcx);
                if regs_clone.rcx == 0 {
                    res.ecx.set_bit(0x5, false); // clear waitpkg
                }

                res
            }
            LEAF_PROCESSOR_EXTENDED_STATE_ENUMERATION => {
                self.load_guest_xstate();
                let res = cpuid!(regs_clone.rax, regs_clone.rcx);
                self.load_host_xstate();

                res
            }
            LEAF_HYPERVISOR_INFO => CpuIdResult {
                eax: LEAF_HYPERVISOR_FEATURE,
                ebx: vendor_regs[0],
                ecx: vendor_regs[1],
                edx: vendor_regs[2],
            },
            LEAF_HYPERVISOR_FEATURE => CpuIdResult {
                eax: 0,
                ebx: 0,
                ecx: 0,
                edx: 0,
            },
            _ => cpuid!(regs_clone.rax, regs_clone.rcx),
        };

        trace!(
            "VM exit: CPUID({:#x}, {:#x}): {:?}",
            regs_clone.rax,
            regs_clone.rcx,
            res
        );

        let regs = self.regs_mut();
        regs.rax = res.eax as _;
        regs.rbx = res.ebx as _;
        regs.rcx = res.ecx as _;
        regs.rdx = res.edx as _;
        self.advance_rip(VM_EXIT_INSTR_LEN_CPUID)?;

        Ok(())
    }

    fn handle_xsetbv(&mut self) -> HyperResult {
        const XCR_XCR0: u64 = 0;
        const VM_EXIT_INSTR_LEN_XSETBV: u8 = 3;

        let index = self.guest_regs.rcx.get_bits(0..32);
        let value = self.guest_regs.rdx.get_bits(0..32) << 32 | self.guest_regs.rax.get_bits(0..32);

        // TODO: get host-supported xcr0 mask by cpuid and reject any guest-xsetbv violating that
        if index == XCR_XCR0 {
            Xcr0::from_bits(value)
                .and_then(|x| {
                    if !x.contains(Xcr0::XCR0_FPU_MMX_STATE) {
                        return None;
                    }

                    if x.contains(Xcr0::XCR0_AVX_STATE) && !x.contains(Xcr0::XCR0_SSE_STATE) {
                        return None;
                    }

                    if x.contains(Xcr0::XCR0_BNDCSR_STATE) ^ x.contains(Xcr0::XCR0_BNDREG_STATE) {
                        return None;
                    }

                    if x.contains(Xcr0::XCR0_OPMASK_STATE)
                        || x.contains(Xcr0::XCR0_ZMM_HI256_STATE)
                        || x.contains(Xcr0::XCR0_HI16_ZMM_STATE)
                    {
                        if !x.contains(Xcr0::XCR0_AVX_STATE)
                            || !x.contains(Xcr0::XCR0_OPMASK_STATE)
                            || !x.contains(Xcr0::XCR0_ZMM_HI256_STATE)
                            || !x.contains(Xcr0::XCR0_HI16_ZMM_STATE)
                        {
                            return None;
                        }
                    }

                    Some(x)
                })
                .ok_or(HyperError::InvalidParam)
                .and_then(|x| {
                    self.xstate.guest_xcr0 = x.bits();
                    self.advance_rip(VM_EXIT_INSTR_LEN_XSETBV)
                })
        } else {
            // xcr0 only
            Err(crate::HyperError::NotSupported)
        }
    }

    fn load_guest_xstate(&mut self) {
        unsafe {
            xcr0_write(Xcr0::from_bits_unchecked(self.xstate.guest_xcr0));
            Msr::IA32_XSS.write(self.xstate.guest_xss);
        }
    }

    fn load_host_xstate(&mut self) {
        unsafe {
            xcr0_write(Xcr0::from_bits_unchecked(self.xstate.host_xcr0));
            Msr::IA32_XSS.write(self.xstate.host_xss);
        }
    }
}

impl<H: HyperCraftHal> Drop for VmxVcpu<H> {
    fn drop(&mut self) {
        unsafe { vmx::vmclear(self.vmcs.phys_addr() as u64).unwrap() };
        info!("[HV] dropped VmxVcpu(vmcs: {:#x})", self.vmcs.phys_addr());
    }
}

fn get_tr_base(tr: SegmentSelector, gdt: &DescriptorTablePointer<u64>) -> u64 {
    let index = tr.index() as usize;
    let table_len = (gdt.limit as usize + 1) / core::mem::size_of::<u64>();
    let table = unsafe { core::slice::from_raw_parts(gdt.base, table_len) };
    let entry = table[index];
    if entry & (1 << 47) != 0 {
        // present
        let base_low = entry.get_bits(16..40) | entry.get_bits(56..64) << 24;
        let base_high = table[index + 1] & 0xffff_ffff;
        base_low | base_high << 32
    } else {
        // no present
        0
    }
}

impl<H: HyperCraftHal> Debug for VmxVcpu<H> {
    fn fmt(&self, f: &mut Formatter) -> Result {
        (|| -> HyperResult<Result> {
            Ok(f.debug_struct("VmxVcpu")
                .field("guest_regs", &self.guest_regs)
                .field("rip", &VmcsGuestNW::RIP.read()?)
                .field("rsp", &VmcsGuestNW::RSP.read()?)
                .field("rflags", &VmcsGuestNW::RFLAGS.read()?)
                .field("cr0", &VmcsGuestNW::CR0.read()?)
                .field("cr3", &VmcsGuestNW::CR3.read()?)
                .field("cr4", &VmcsGuestNW::CR4.read()?)
                .field("cs", &VmcsGuest16::CS_SELECTOR.read()?)
                .field("fs_base", &VmcsGuestNW::FS_BASE.read()?)
                .field("gs_base", &VmcsGuestNW::GS_BASE.read()?)
                .field("tss", &VmcsGuest16::TR_SELECTOR.read()?)
                .finish())
        })()
        .unwrap()
    }
}
