"""Expands tabs into HTML."""

import dataclasses
import json
import os
import pathlib
import sys


@dataclasses.dataclass
class MarkdownFile:
    name: str
    path: str
    section: str
    is_index: bool


def find_all_markdowns(root: str) -> dict[str, str]:
    """Find all markdown files in the given directory."""
    markdowns = {}
    for root, dirs, files in os.walk(root):
        for file in files:
            if file.endswith(".md"):
                markdowns[file] = os.path.join(root, file)

    return markdowns


def output_chapter(path: str, number: int):
    """Read the markdown file and output the chapter."""
    file_content = pathlib.Path(path).read_text()

    # parse the title from the non-empty line
    name = ""

    for line in file_content.splitlines():
        if line.strip():
            name = line.strip("# ")
            break

    chapter = {
        "name": name,
        "content": file_content,
        "number": number,
        "sub_items": [],
        "parent_names": [],
        "path": f"{path[2:]}",
        "source_path": f"../../{path[2:]}",
    }

    return chapter


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "supports":
            sys.exit(0)

    context, book = json.load(sys.stdin)

    config = context["config"]["preprocessor"]["adr"]

    for section in config["section"]:
        markdowns = find_all_markdowns(section["path"])
        title = section["title"]
        book["sections"].append(
            {
                "PartTitle": title,
            }
        )

        # we process the readme first to make it the index
        if "README.md" in markdowns:
            p = markdowns.pop("README.md")
            chapter = output_chapter(p, [1])
            chapter["name"] = title
            book["sections"].append(
                {
                    "Chapter": chapter,
                }
            )

        for idx, path in enumerate(sorted(markdowns.keys())):
            chapter = output_chapter(markdowns[path], [idx + 1])
            book["sections"].append(
                {
                    "Chapter": chapter,
                }
            )

    print(json.dumps(book))
