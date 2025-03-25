import type { NextApiRequest, NextApiResponse } from 'next';
import { loadConfig } from '../../server/config';

export default function handler(req: NextApiRequest, res: NextApiResponse) {
    try {
        const config = loadConfig();
        res.status(200).json(config);
    } catch (e) {
        res.status(500).json({ error: 'Failed to load config' });
    }
}