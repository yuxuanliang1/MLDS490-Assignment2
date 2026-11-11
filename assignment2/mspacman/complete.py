"""Sequential post-main validations and report/audit; does not reselect the primary."""
from pathlib import Path
import sys
from .validate import execute


def main():
    root=Path('assignment2/results/mspacman');ledger=root/'completion_commands.jsonl'
    commands=[['-m','assignment2.mspacman.validate','short-suite','--steps','100000'],
        ['-m','assignment2.mspacman.validate','same-seed-prefix'],
        ['-m','assignment2.mspacman.smoke','--root',str(root/'smoke_recovery_extended'),'--rollback-test'],
        ['-m','assignment2.mspacman.report'],['-m','assignment2.mspacman.audit']]
    for args in commands:execute(args,ledger)


if __name__=='__main__':main()
