import os
from dotenv import load_dotenv
load_dotenv()

from app import generate_email_reply, generate_fba_reply

DIVIDER = "=" * 65

EMAIL_TESTS = [
     ("initial_outreach_garden_4", "200k partnership opportunity email to EGO Power, sales manager, garden and tools"),
]

FBA_TESTS = [
    "What is the Buy Box and how do I win it?",
    "How do I validate demand for a wholesale product before ordering?",
]


def run_email_tests():
    print(f"\n{DIVIDER}")
    print("  EMAIL ROUTING TESTS")
    print(DIVIDER)
    for expected_id, message in EMAIL_TESTS:
        print(f"\nExpected route : {expected_id}")
        print(f"Input          : {message}")
        print("-" * 65)
        result = generate_email_reply(message)
        print(result)
        print()


def run_fba_tests():
    print(f"\n{DIVIDER}")
    print("  FBA Q&A TESTS")
    print(DIVIDER)
    for question in FBA_TESTS:
        print(f"\nQuestion : {question}")
        print("-" * 65)
        result = generate_fba_reply("", question)
        print(result)
        print()


def run_confuse_tests():
    confuse_cases = [e for e in EMAIL_TESTS if " vs " in e[0] or " or " in e[0]]
    print(f"\n{DIVIDER}")
    print("  CONFUSE / EDGE CASE TESTS")
    print(DIVIDER)
    for expected_id, message in confuse_cases:
        print(f"\nAmbiguous between : {expected_id}")
        print(f"Input             : {message}")
        print("-" * 65)
        result = generate_email_reply(message)
        print(result)
        print()


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode == "email":
        run_email_tests()
    elif mode == "fba":
        run_fba_tests()
    elif mode == "confuse":
        run_confuse_tests()
    else:
        run_email_tests()
        run_fba_tests()
