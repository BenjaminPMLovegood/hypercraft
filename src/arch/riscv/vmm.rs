use alloc::vec::Vec;
use riscv::register::time;

use crate::{
    arch::csrs::{traps, RiscvCsrTrait, CSR},
    GuestPageTableTrait, HyperCraftHal,
};

use super::VM;
/// virtual machine manager
pub struct VMM<H: HyperCraftHal, G: GuestPageTableTrait> {
    vm_list: Vec<VM<H, G>>,
}

fn get_time() -> u64 {
    time::read() as u64
}

// TODO: 確定 qemu 的時鐘頻率
const TIME_SLICE: u64 = 10_0000;

fn set_next_trigger() {
    sbi_rt::set_timer(get_time() + TIME_SLICE);
}

impl<H: HyperCraftHal, G: GuestPageTableTrait> VMM<H, G> {
    /// 創建新的 VMM ，底下無任何虛擬機
    pub fn new() -> Self {
        VMM { vm_list: vec![] }
    }
    /// 將虛擬機加入 VMM
    pub fn add_vm(&mut self, vm: VM<H, G>) {
        self.vm_list.push(vm);
    }
    /// 在 hart_id 上執行 VMM 管理的所有虛擬機
    pub fn run(&mut self, hart_id: usize) {
        let vm_number = self.vm_list.len();
        assert_ne!(vm_number, 0);

        info!("vmm run cpu{}", hart_id);

        // CSR.sie
        //     .read_and_set_bits(traps::interrupt::SUPERVISOR_TIMER);

        let mut id = 0;
        loop {
            debug!("執行虛擬機 {}", id);

            // set_next_trigger();

            self.vm_list[id].run(0);

            id = (id + 1) % vm_number;
        }
    }
}
