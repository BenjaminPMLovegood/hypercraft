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
const TIME_SLICE: u64 = 20_0000;

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
        debug!("虛擬機數量 {}", vm_number);

        info!("vmm run cpu{}", hart_id);

        CSR.sie
            .read_and_set_bits(traps::interrupt::SUPERVISOR_TIMER);

        let mut id = 0;
        let mut selected_vm_id_for_input = 0;
        info!("由虛擬機 {} 控制獲取輸入", selected_vm_id_for_input);
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
                    // XXX: 爲什麼需要阻斷 host 的 sie ？
                    // nimbos 的時鐘中斷處理器也會 setTimer ，但並不是根據當前時間加上一個時間片
                    // 而是根據上一次 setTimer(timer) 的 timer 加上一個時間片
                    // 如果它 timer 加上時間片的增長速度比真實時間增長得還要慢，會發生以下循環：
                    // 1.  虛擬機呼叫 SetTimer ，hypervisor 呼叫的 sbi_rt::set_timer 的時間點會比當下時間還要早
                    // 2.  若不阻斷 sie.STIE ，一進入虛擬機，hypervisor 馬上被時鐘中斷
                    // 3.  掉入 VmmTrap::TimerInterruptEmultaion ，也就是當前代碼
                    // 4.  由於當前時間比虛擬機 timer 晚，注入 hvip.STIP
                    // 5.a 若 hypervisor 時間片還沒到，timer 沒有重設，回到 2.
                    // 5.b 若 hypervisor 時間片到了，timer 終於被重設到未來的時間點，進入虛擬機之後不再立刻觸發 hypervisor 的中斷
                    //     但由於剛注入過 hvip.STIP ，虛擬機進入其時鐘中斷處理器，重新呼叫 setTimer ，回到 1.
                    // 在整個循環中，虛擬機都沒辦法做到事。
                    //
                    // 若阻斷 sie.STIE ，則 nimbos 虛擬機不會頻繁跳入 hypervisor ，其 timer 追上真實時間的機會就增加了。
                    //
                    // 那總是在進入虛擬機之前關閉 sie.STIE 有沒有壞處呢？有的，如果虛擬機本身不會主動 SetTimer
                    // sie.STIE 永遠不會被開啓，從而導致時鐘中斷無法中斷虛擬機的執行，單個虛擬機可以獨佔 CPU 資源
                    // 所以下面這行到底要不要開啓，還是個兩難
                    CSR.sie
                        .read_and_clear_bits(traps::interrupt::SUPERVISOR_TIMER);

                    let time = get_time();
                    for vm in &mut self.vm_list {
                        if time > vm.get_timer() {
                            // TODO: 僅注入到該 vm
                            // debug!("注入 hvip.STIP");
                            CSR.hvip
                                .read_and_set_bits(traps::interrupt::VIRTUAL_SUPERVISOR_TIMER);
                        }
                    }
                    // 現在時間已經超出時間片，切換虛擬機
                    if time > self.switch_vm_timer {
                        debug!("現在時間 {}，時間片到期時間 {}", time, self.switch_vm_timer);
                        debug!("切換虛擬機");

                        // 讀取 console 所有可讀字元
                        loop {
                            let c = sbi_rt::legacy::console_getchar();
                            if c == usize::MAX {
                                break;
                            } else if c == 96 {
                                selected_vm_id_for_input =
                                    (selected_vm_id_for_input + 1) % vm_number;
                                info!("由虛擬機 {} 控制獲取輸入", selected_vm_id_for_input);
                            } else {
                                info!("注入虛擬機 {} 輸入 {}", selected_vm_id_for_input, c);
                                self.vm_list[selected_vm_id_for_input].add_char_to_input_buffer(c);
                            }
                        }

                        // 設定下個時間片
                        self.set_switch_vm_timer();

                        // 執行下一個虛擬機
                        id = (id + 1) % vm_number;
                    }
                }
            }
        }
    }
}
