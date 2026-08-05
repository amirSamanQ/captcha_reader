"""Example: log in to a Sahra OMS account using the SDK.

Run it and pass your own credentials. Never commit real credentials to git;
read them from environment variables or type them in at runtime instead.

    export OMS_USERNAME="your_bourse_code"
    export OMS_PASSWORD="your_password"
    python examples/login_example.py
"""

import os

from captcha_reader import login


def main():
    username = os.environ.get("OMS_USERNAME") or input("Username: ")
    password = os.environ.get("OMS_PASSWORD") or input("Password: ")

    # broker="bbi" targets https://identity-bbi.ephoenix.ir (Sahra OMS),
    # which is what the bundled sample model was trained for.
    result = login(username, password, broker="bbi", max_attempts=15)

    print("token     :", (result["token"] or "")[:40], "...")
    print("session_id:", result["session_id"])
    print("cookie    :", result["cookie"][:60], "...")


if __name__ == "__main__":
    main()
