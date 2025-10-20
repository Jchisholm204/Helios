#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/mm.h>
#include <linux/sched/mm.h>
#include <linux/pid.h>
#include <linux/rcupdate.h>
#include <linux/slab.h>

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Example");
MODULE_DESCRIPTION("MMU Notifier for a single process");

// Target PID (set via insmod)
static int target_pid = -1;
module_param(target_pid, int, 0444);
MODULE_PARM_DESC(target_pid, "PID of the process to monitor");

// Our notifier structure
struct my_mmu_notifier {
    struct mmu_notifier mmu_notifier;
    struct mm_struct *mm;
};

// Callback function when memory is invalidated
static void my_invalidate_range_start(struct mmu_notifier *mn,
                                      struct mm_struct *mm,
                                      unsigned long start, unsigned long end)
{
    pr_info("[MMU Notifier] PID %d memory invalidated: 0x%lx - 0x%lx\n",
            task_pid_nr(current), start, end);
}

static const struct mmu_notifier_ops my_ops = {
    .invalidate_range_start = my_invalidate_range_start,
};

static struct my_mmu_notifier *notifier;

static int __init mmu_notifier_init(void)
{
    struct task_struct *task;

    if (target_pid < 0) {
        pr_err("Please provide a valid PID via target_pid parameter\n");
        return -EINVAL;
    }

    rcu_read_lock();
    task = pid_task(find_vpid(target_pid), PIDTYPE_PID);
    if (!task) {
        rcu_read_unlock();
        pr_err("No such process with PID %d\n", target_pid);
        return -ESRCH;
    }

    // Get mm_struct reference
    notifier = kzalloc(sizeof(*notifier), GFP_KERNEL);
    if (!notifier) {
        rcu_read_unlock();
        return -ENOMEM;
    }

    notifier->mm = get_task_mm(task);
    rcu_read_unlock();

    notifier->mmu_notifier.ops = &my_ops;

    // Register MMU notifier
    if (mmu_notifier_register(&notifier->mmu_notifier, notifier->mm)) {
        pr_err("Failed to register mmu_notifier\n");
        mmput(notifier->mm);
        kfree(notifier);
        return -EFAULT;
    }

    pr_info("MMU Notifier registered for PID %d\n", target_pid);
    return 0;
}

static void __exit mmu_notifier_exit(void)
{
    if (!notifier)
        return;

    mmu_notifier_unregister(&notifier->mmu_notifier, notifier->mm);
    mmput(notifier->mm);
    kfree(notifier);

    pr_info("MMU Notifier unregistered\n");
}

module_init(mmu_notifier_init);
module_exit(mmu_notifier_exit);

