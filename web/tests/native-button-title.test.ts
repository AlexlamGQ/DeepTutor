import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import test from "node:test";
import ts from "typescript";

type TitleDebt = Record<string, string[]>;

const root = process.cwd();
// Existing native hints remain in this debt snapshot. Deleting one needs no
// fixture update; adding one fails even when another file was cleaned up.
const baseline: TitleDebt = JSON.parse(
  readFileSync(path.join(root, "tests/native-button-title-baseline.json"), "utf8"),
);

function sourceFiles(directory: string): string[] {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const file = path.join(directory, entry.name);
    if (entry.isDirectory()) return sourceFiles(file);
    return entry.name.endsWith(".tsx") ? [file] : [];
  });
}

function attribute(
  node: ts.JsxOpeningElement | ts.JsxSelfClosingElement,
  name: string,
): ts.JsxAttribute | undefined {
  return node.attributes.properties.find(
    (property): property is ts.JsxAttribute =>
      ts.isJsxAttribute(property) && property.name.getText() === name,
  );
}

function signature(
  node: ts.JsxOpeningElement | ts.JsxSelfClosingElement,
  source: ts.SourceFile,
): string {
  // Ignore styling and line movement. The hint and action identify a legacy
  // control; adding another control with the same pair still exceeds its
  // allowed occurrence count.
  const details = ["title", "onClick", "aria-label"]
    .map((name) => attribute(node, name)?.initializer?.getText(source) ?? "")
    .map((value) => value.replace(/\s+/g, " ").trim())
    .join("|");
  return createHash("sha256").update(details).digest("hex").slice(0, 16);
}

test("new native button titles must use the shared Tooltip", () => {
  const violations: string[] = [];
  for (const sourceRoot of [
    "app",
    "components",
    "context",
    "features",
    "hooks",
    "lib",
    "shared",
  ]) {
    for (const file of sourceFiles(path.join(root, sourceRoot))) {
      const relative = path.relative(root, file).split(path.sep).join("/");
      const source = ts.createSourceFile(
        file,
        readFileSync(file, "utf8"),
        ts.ScriptTarget.Latest,
        true,
        ts.ScriptKind.TSX,
      );
      const allowed = [...(baseline[relative] ?? [])];
      const visit = (node: ts.Node) => {
        if (
          (ts.isJsxOpeningElement(node) || ts.isJsxSelfClosingElement(node)) &&
          node.tagName.getText(source) === "button" &&
          attribute(node, "title")
        ) {
          const hash = signature(node, source);
          const index = allowed.indexOf(hash);
          if (index >= 0) allowed.splice(index, 1);
          else {
            const line = source.getLineAndCharacterOfPosition(node.getStart(source)).line + 1;
            violations.push(`${relative}:${line}`);
          }
        }
        ts.forEachChild(node, visit);
      };
      visit(source);
    }
  }
  assert.deepEqual(
    violations,
    [],
    "Replace new <button title> hints with @/shared/ui/Tooltip and an accessible button name",
  );
});
