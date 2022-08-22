import json
import re
import shutil
import subprocess
import warnings
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Sequence, Any, Union

from github import Github
from github.ContentFile import ContentFile
from pydantic import BaseModel


@dataclass(frozen=True)
class PrudensRule:
    """A class to represent a Prudens rule and its components."""

    body: tuple[str, ...]
    head: str
    added_as: str = ""
    active: bool = True

    @classmethod
    def from_string(cls, rule_string: str, added_as: str = "", active: bool = True):
        assert all([p in rule_string for p in [" :: ", " implies "]])
        ctx, head = rule_string.strip().rstrip(";").split(" :: ")[1].split(" implies ")
        return cls(
            body=tuple(ctx.split(", ")),
            head=head.strip(),
            added_as=added_as,
            active=active,
        )

    @classmethod
    def from_prudens_rule_json_object(
        cls, prudens_rule_json_object: dict[str, Union[str, list, dict]]
    ):
        """
        Parses a Prudens rule from JSON as described here https://github.com/VMarkos/prudens-js#rules. The `name` field
        is ignored.
        """
        assert all(k in prudens_rule_json_object for k in ["name", "body", "head"])
        return cls(
            body=tuple(
                PrudensLiteral.parse_obj(lit).to_string()
                for lit in prudens_rule_json_object["body"]
            ),
            head=PrudensLiteral.parse_obj(prudens_rule_json_object["head"]).to_string(),
        )

    def to_string(self, rule_name=None) -> str:
        if rule_name is None:
            rule_name = "R"
        elif isinstance(rule_name, int):
            rule_name = f"R{rule_name}"

        return " ".join(
            [
                rule_name,
                "::",
                ", ".join(self.body),
                "implies",
                f"{self.head};",
            ]
        )

    def __str__(self) -> str:
        return self.to_string()

    def __eq__(self, obj: object) -> bool:
        """
        Checks equality between two PrudensRule instances. NOTE that two rules with a body of the same literals, but
        in a different order, ARE considered equal, e.g.:

        `PrudensRule(body=("a", "b"), head="c") == PrudensRule(body=("b", "a"), head="c")`
        """
        return (
            isinstance(obj, PrudensRule)
            and set(self.body) == set(obj.body)
            and self.head == obj.head
        )


@dataclass(frozen=True)
class PrudensKnowledgeBase:
    rules: tuple[PrudensRule, ...]

    def is_empty(self) -> bool:
        return len(self.rules) == 0

    def is_not_empty(self) -> bool:
        return not self.is_empty()

    def number_of_active_rules(self) -> int:
        return len([r for r in self.rules if r.active])

    def get_full_context(self) -> list[str]:
        def natural_sort_key(s, _nsre=re.compile(r"(\d+)")):
            """
            Can be used to sort strings in natural order (taking into account numbers in strings). Copied from
            https://stackoverflow.com/a/16090640/.
            """
            return [int(t) if t.isdigit() else t.lower() for t in _nsre.split(s)]

        full_context = set()
        for rule in self.rules:
            for literal in rule.body:
                if literal not in ["true", "empty"]:
                    full_context.add(literal.replace("-", ""))
        return sorted(full_context, key=natural_sort_key)

    def copy(self) -> "PrudensKnowledgeBase":
        """
        Creates a copy of an instance of this class.
        """
        return PrudensKnowledgeBase(rules=tuple(self.rules))

    def to_string(self, sep: str = "\n") -> str:
        return sep.join(
            [
                "@KnowledgeBase",
                *[
                    rule.to_string(i)
                    for i, rule in enumerate(self.rules, start=1)
                    if rule.active
                ],
            ]
        )

    @classmethod
    def from_string(cls, kb_string: str):
        assert kb_string.strip().startswith(
            "@KnowledgeBase"
        ), "KB string must begin with '@KnowledgeBase'!"

        rules_str = (
            kb_string.strip()
            .replace("\n", " ")
            .lstrip("@KnowledgeBase")
            .strip()
            .split(";")
        )

        rules_str = [r.strip() for r in rules_str if r]

        return cls(rules=tuple(map(PrudensRule.from_string, rules_str)))

    def to_prudens_kb_json_object(self):
        rules_converted = [
            {
                "name": f"R{i}",
                "body": [to_prudens_literal(lit) for lit in rule.body],
                "head": to_prudens_literal(rule.head),
            }
            for i, rule in enumerate(self.rules)
            if rule.active
        ]

        return {
            "type": "output",
            "kb": rules_converted,
            # "code": "",
            "imports": "",
            "warnings": [],
            # "constraints": "",
        }

    def __str__(self) -> str:
        return self.to_string()

    def __eq__(self, obj: object) -> bool:
        return isinstance(obj, PrudensKnowledgeBase) and self.rules == obj.rules


@cache
def to_prudens_literal(lit: str) -> dict[str, Union[bool, int, str]]:
    """
    Converts the provided literal to a dict object used to communicate with Prudens.

    Note: Only the fields name and sign are implemented.
    """
    return PrudensLiteral(
        name=lit.lstrip("-"),
        sign="-" not in lit,
        isJS=False,
        isEquality=False,
        isInequality=False,
        isAction=False,
        arity=0,
    ).dict()


class PrudensLiteral(BaseModel):
    name: str
    sign: bool
    isJS: bool
    isEquality: bool
    isInequality: bool
    isAction: bool
    # args: list
    arity: int

    def to_string(self) -> str:
        # when sign is True -> positive literal, and the reverse
        return f"{'-' if not self.sign else ''}{self.name}"


def get_prudens_source(sha: str = "3852ab4a3d6e5f3537af77fa0c17a3c198920877") -> Path:
    """
    Looks for the Prudens source code file, if it does not exist it's downloaded from GitHub. Returns the path to the
    file.
    """
    this_script_dir = Path(__file__).resolve().parent
    prudens_js_source_path = Path(this_script_dir, f"prudens_js_source_{sha}.js")

    if not prudens_js_source_path.exists():
        repo = Github().get_repo("VMarkos/prudens-js")

        # here the contents of the repo are retrieved using get_git_tree, instead of the recommended method, see
        # https://pygithub.readthedocs.io/en/latest/examples/Repository.html#get-all-of-the-contents-of-the-repository-recursively
        # to avoid typing issues with mypy
        # the git tree looks like this: https://api.github.com/repos/VMarkos/prudens-js/git/trees/main?recursive=1
        git_tree = repo.get_git_tree(sha=sha, recursive=True).tree

        all_js_files: list[ContentFile] = []
        for elem in git_tree:
            if elem.type == "blob" and elem.path.endswith(".js"):
                content_file = repo.get_contents(elem.path, ref=sha)
                if isinstance(content_file, ContentFile):
                    all_js_files.append(content_file)

        # get contents of required files
        source = []
        for js_file in all_js_files:
            if (
                "eventHandling.js" not in js_file.name
                and "kb_generator" not in js_file.path
            ):
                source.extend(
                    [
                        f"/////////// {js_file.name} ///////////",
                        js_file.decoded_content.decode("utf-8"),
                        "\n",
                    ]
                )

        with Path(this_script_dir, "prudens_bridge.js").open("r") as f:
            bridge_code = f.read()
        source.append(bridge_code)

        with prudens_js_source_path.open("w+") as f:
            f.write("\n".join(source))

    return prudens_js_source_path


def simplify_prudens_rule(rule: PrudensRule) -> list[PrudensRule]:
    """
    Generates all possible simplifications of the provided Prudens rule, when 1 literal is removed from the rule's body.
    """
    if len(rule.body) == 1:
        return []
    else:
        simplified_rules = []
        for literal in rule.body:
            new_body = list(rule.body)
            new_body.remove(literal)
            simplified_rules.append(
                PrudensRule(body=tuple(new_body), head=rule.head, added_as="Simplify")
            )

        return simplified_rules


@cache
def cmd_exists(cmd: str) -> bool:
    """
    Checks if the provided command exists on PATH (based on https://stackoverflow.com/a/28909933).
    """
    return shutil.which(cmd) is not None


def run_prudens(
    prudens_inputs: dict[str, Sequence[Any]], source_file_path: Path
) -> list[list]:
    node_cmd = "node"

    if not cmd_exists(node_cmd):
        raise AssertionError("Node JS must be installed and added to PATH.")

    res = subprocess.run(
        args=[node_cmd, source_file_path.absolute()],
        input=json.dumps(prudens_inputs),
        capture_output=True,
        text=True,
    )

    if len(res.stderr.strip()) > 0:
        warnings.warn(f"Error message from Node JS:\n{res.stderr}")

    return [json.loads(r) for r in res.stdout.splitlines()]
