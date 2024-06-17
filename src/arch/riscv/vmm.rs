use alloc::vec::Vec;
use riscv::register::time;

use crate::{
    arch::{
        csrs::{traps, RiscvCsrTrait, CSR},
        vmm_trap::VmmTrap,
    },
    GuestPageTableTrait, HyperCraftHal,
};

use super::VM;
/// virtual machine manager
pub struct VMM<H: HyperCraftHal, G: GuestPageTableTrait> {
    vm_list: Vec<VM<H, G>>,
    switch_vm_timer: u64,
}

fn get_time() -> u64 {
    time::read() as u64
}

// TODO: 確定 qemu 的時鐘頻率
const TIME_SLICE: u64 = 200_0000;

impl<H: HyperCraftHal, G: GuestPageTableTrait> VMM<H, G> {
    /// 創建新的 VMM ，底下無任何虛擬機
    pub fn new() -> Self {
        VMM {
            vm_list: vec![],
            switch_vm_timer: u64::MAX,
        }
    }
    /// 將虛擬機加入 VMM
    pub fn add_vm(&mut self, vm: VM<H, G>) {
        self.vm_list.push(vm);
    }
    fn set_switch_vm_timer(&mut self) {
        self.switch_vm_timer = get_time() + TIME_SLICE;
        CSR.sie
            .read_and_set_bits(traps::interrupt::SUPERVISOR_TIMER);
        sbi_rt::set_timer(self.switch_vm_timer);
    }
    /// 在 hart_id 上執行 VMM 管理的所有虛擬機
    pub fn run(&mut self, hart_id: usize) {
        let vm_number = self.vm_list.len();
        assert_ne!(vm_number, 0);

        info!("vmm run cpu{}", hart_id);

        CSR.sie
            .read_and_set_bits(traps::interrupt::SUPERVISOR_TIMER);

        let mut id = 0;
        self.set_switch_vm_timer();
        loop {
            // debug!("執行虛擬機 {}", id);

            let vmm_trap = self.vm_list[id].run(0);

            match vmm_trap {
                VmmTrap::SetTimer(timer) => {
                    CSR.sie
                        .read_and_set_bits(traps::interrupt::SUPERVISOR_TIMER);

                    // let time = get_time();
                    // debug!("虛擬機設定時鐘 {}", timer);
                    // debug!("現在時鍾 {}", time);
                    // TODO: 僅清除該 vm 的 hvip bit
                    CSR.hvip
                        .read_and_clear_bits(traps::interrupt::VIRTUAL_SUPERVISOR_TIMER);
                    sbi_rt::set_timer(timer);
                }
                VmmTrap::TimerInterruptEmulation => {
                    CSR.sie
                        .read_and_clear_bits(traps::interrupt::SUPERVISOR_TIMER);

                    let time = get_time();
                    for vm in &mut self.vm_list {
                        if time > vm.get_timer() {
                            // TODO: 僅注入到該 vm
                            // debug!("注入 hvip.stip");
                            CSR.hvip
                                .read_and_set_bits(traps::interrupt::VIRTUAL_SUPERVISOR_TIMER);
                        }
                    }
                    // 現在時間已經超出時間片，切換虛擬機
                    if time > self.switch_vm_timer {
                        debug!("現在時間 {}，時間片到期時間 {}", time, self.switch_vm_timer);
                        debug!("切換虛擬機");
                        self.set_switch_vm_timer();
                        id = (id + 1) % vm_number;
                    }
                }
            }
        }
    }
}
