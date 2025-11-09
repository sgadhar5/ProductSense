import json

# read exported browser cookies
with open("cookies.json", "r") as f:
    browser_cookies = json.load(f)

# convert list → dict
cookie_dict = {cookie["name"]: cookie["value"] for cookie in browser_cookies}

# save Twikit-compatible format
with open("cookies_twikit.json", "w") as f:
    json.dump(cookie_dict, f, indent=2)

print("✅ Converted cookies.json → cookies_twikit.json")
