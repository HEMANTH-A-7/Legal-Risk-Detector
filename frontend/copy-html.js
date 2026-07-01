import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const source = path.join(__dirname, '../static/dist/index.html');
const dest = path.join(__dirname, '../templates/index.html');

try {
  // Ensure the target directory exists
  const destDir = path.dirname(dest);
  if (!fs.existsSync(destDir)) {
    fs.mkdirSync(destDir, { recursive: true });
  }

  // Copy file
  fs.copyFileSync(source, dest);
  console.log(`Successfully copied ${source} to ${dest}`);
} catch (err) {
  console.error('Error copying index.html:', err);
  process.exit(1);
}
