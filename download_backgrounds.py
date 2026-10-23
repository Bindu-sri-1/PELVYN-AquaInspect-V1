import requests
from pathlib import Path
from PIL import Image
from io import BytesIO
import time

BASE = Path("background_images")
BASE.mkdir(exist_ok=True)

SEARCHES = {
    "empty_water": [
        "pond water surface", "lake open water", "river water surface",
        "water reflections no plants"
    ],
    "mud_rocks": [
        "muddy pond shoreline", "rocks beside water",
        "muddy river bank", "wet soil stones"
    ],
    "grass_reeds": [
        "pond reeds grass", "wetland grass", "cattails marsh",
        "reeds beside river"
    ],
    "other_vegetation": [
        "water lily pond", "duckweed pond", "aquatic plants pond",
        "pond vegetation"
    ],
}

TARGET = 50
API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "PELVYNAquaInspect/1.0 (educational project)"}

for category, queries in SEARCHES.items():
    folder = BASE / category
    folder.mkdir(parents=True, exist_ok=True)

    existing = list(folder.glob("*.jpg"))
    count = len(existing)
    seen = set()
    print(f"\n{category}: existing {count}, target {TARGET}")

    for query in queries:
        if count >= TARGET:
            break

        try:
            response = requests.get(
                API,
                params={
                    "action": "query",
                    "generator": "search",
                    "gsrsearch": f"filetype:bitmap {query}",
                    "gsrnamespace": 6,
                    "gsrlimit": 50,
                    "prop": "imageinfo",
                    "iiprop": "url|extmetadata",
                    "iiurlwidth": 1000,
                    "format": "json",
                },
                headers=HEADERS,
                timeout=30,
            )
            response.raise_for_status()
            pages = response.json().get("query", {}).get("pages", {})

            for page in pages.values():
                if count >= TARGET:
                    break

                title = page.get("title", "")
                if title in seen:
                    continue
                seen.add(title)

                info_list = page.get("imageinfo", [])
                if not info_list:
                    continue
                info = info_list[0]
                url = info.get("thumburl") or info.get("url")
                if not url:
                    continue

                try:
                    r = requests.get(url, headers=HEADERS, timeout=25)
                    r.raise_for_status()
                    img = Image.open(BytesIO(r.content)).convert("RGB")
                    if img.width < 300 or img.height < 200:
                        continue

                    filename = f"{category}_{count + 1:03d}.jpg"
                    img.save(folder / filename, "JPEG", quality=90)
                    count += 1

                    metadata = info.get("extmetadata", {})
                    license_name = metadata.get("LicenseShortName", {}).get("value", "Check license")
                    source = info.get("descriptionurl", "")
                    with open(folder / "sources.txt", "a", encoding="utf-8") as log:
                        log.write(f"{filename}\nTitle: {title}\nSource: {source}\nLicense: {license_name}\n\n")

                    print(f"Saved {count}/{TARGET}: {filename}")
                    time.sleep(0.2)

                except Exception as e:
                    print("Skipped image:", str(e)[:100])

        except Exception as e:
            print(f"Search failed for '{query}':", str(e)[:150])

    print(f"Finished {category}: {count} images available.")

print("\nDownload process finished.")
print("IMPORTANT: Manually inspect every image before training.")
print("Do not use images containing water hyacinth as empty negative examples.")
