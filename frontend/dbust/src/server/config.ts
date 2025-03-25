import yaml from 'js-yaml';
import path from 'path';
import fs from 'fs';

interface Config {
    apiBaseUrl: string;
}

let config: Config;

export const loadConfig = (): Config => {
    if (!config) {
        try {
            const filePath = path.resolve(process.cwd(), 'config.yaml');
            const fileContents = fs.readFileSync(filePath, 'utf8');
            config = yaml.load(fileContents) as Config;
        } catch (e) {
            console.error(e);
            throw new Error('Failed to load config');
        }
    }
    return config;
};