import argparse


def main():
    parser = argparse.ArgumentParser(prog="sabr", description="SAbR CLI")
    parser.add_argument("-u", action="store_true", help="print in uppercase")
    args = parser.parse_args()

    msg = "hello world"
    if args.u:
        msg = msg.upper()
    print(msg)


if __name__ == "__main__":
    main()
