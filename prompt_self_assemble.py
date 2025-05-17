import sys
import self_assemble


def main():
    if len(sys.argv) < 2:
        print("Usage: python prompt_self_assemble.py '<prompt>'")
        return
    prompt = " ".join(sys.argv[1:])
    self_assemble.prompt_mode(prompt)


if __name__ == "__main__":
    main()
