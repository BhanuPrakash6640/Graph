#!/usr/bin/env node
/**
 * Guard script to enforce AGENTS.md Rule 1:
 * "Every number must come from the API, which reads files in reports/ or experiments/results/."
 *
 * Scans all frontend/app page.tsx files and fails if any decimal literal
 * (e.g. 0.xxx, 0.xx) is hardcoded outside permitted styling/scale props.
 */

const fs = require("fs");
const path = require("path");

const APP_DIR = path.join(__dirname, "..", "app");

function findPageFiles(dir, fileList = []) {
  const files = fs.readdirSync(dir);
  for (const file of files) {
    const fullPath = path.join(dir, file);
    const stat = fs.statSync(fullPath);
    if (stat.isDirectory()) {
      findPageFiles(fullPath, fileList);
    } else if (file === "page.tsx") {
      fileList.push(fullPath);
    }
  }
  return fileList;
}

const pageFiles = findPageFiles(APP_DIR);
let errorsFound = 0;

// Matches decimal literals like 0.8001, 0.072, 0.638
// Negative lookbehind prevents matching tailwind classes like py-0.5, text-[0.8rem]
const DECIMAL_REGEX = /(?<![\w\/-])0\.\d{2,}(?![\w-])/g;

for (const filePath of pageFiles) {
  const relPath = path.relative(path.join(__dirname, ".."), filePath).replace(/\\/g, "/");
  const content = fs.readFileSync(filePath, "utf-8");
  const lines = content.split("\n");

  lines.forEach((line, idx) => {
    const trimmed = line.trim();
    // Skip full comments
    if (trimmed.startsWith("//") || trimmed.startsWith("/*") || trimmed.startsWith("*")) {
      return;
    }

    // Ignore styling-only lines (e.g. cytoscape arrow-scale, opacity, zoom factor)
    if (
      trimmed.includes('"arrow-scale"') ||
      trimmed.includes("zoom()") ||
      trimmed.includes("opacity")
    ) {
      return;
    }

    const matches = line.match(DECIMAL_REGEX);
    if (matches) {
      console.error(
        `❌ [RULE 1 VIOLATION] Hardcoded decimal literal in ${relPath}:${idx + 1}:`
      );
      console.error(`   Line: ${trimmed}`);
      console.error(`   Found: ${matches.join(", ")}`);
      errorsFound++;
    }
  });
}

if (errorsFound > 0) {
  console.error(`\nFailed: Found ${errorsFound} hardcoded decimal literal(s) across page.tsx files.`);
  console.error("All data numbers must be served dynamically from backend API endpoints.\n");
  process.exit(1);
} else {
  console.log(`✔ PASS: Checked ${pageFiles.length} page.tsx files. Zero hardcoded data decimals found.`);
  process.exit(0);
}
