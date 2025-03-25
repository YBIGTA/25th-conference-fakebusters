interface Config {
    apiBaseUrl: string;
}

let config: Config;

export const loadConfig = async () => {
    if (typeof window === 'undefined') {
        throw new Error('loadConfig should only be called on the client side');
    }

    try {
        const response = await fetch('/api/config');
        if (!response.ok) {
            throw new Error('Failed to fetch config');
        }
        config = await response.json();
    } catch (e) {
        console.error(e);
    }
};

export const getConfig = async (): Promise<Config> => {
    if (!config) {
        await loadConfig();
    }
    return config;
};