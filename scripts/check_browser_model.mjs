import {readFileSync} from 'node:fs';
import {predict} from '../portfolio/dist/model.mjs';
const model=JSON.parse(readFileSync(new URL('../portfolio/dist/model.json',import.meta.url),'utf8'));
let input='';for await(const chunk of process.stdin)input+=chunk;
process.stdout.write(JSON.stringify(JSON.parse(input).map(text=>predict(model,text))));
