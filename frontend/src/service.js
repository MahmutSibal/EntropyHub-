import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://127.0.0.1:8000';
const API_KEY = process.env.REACT_APP_API_KEY || 'teknofest-local-dev-key';
const API_TIMEOUT_MS = Number(process.env.REACT_APP_API_TIMEOUT_MS || 15000);
const RETRY_COUNT = Number(process.env.REACT_APP_API_RETRY_COUNT || 2);
const RETRY_DELAY_MS = Number(process.env.REACT_APP_API_RETRY_DELAY_MS || 1000);

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const humanizeApiError = (error) => {
  if (error.code === 'ECONNABORTED' || error.message?.toLowerCase().includes('timeout')) {
    return {
      type: 'timeout',
      status: 0,
      message: 'API zaman asimina ugradi. Lutfen tekrar deneyin.',
      retryAfterSec: null,
    };
  }

  const status = error.response?.status || 0;
  const retryAfter = error.response?.headers?.['retry-after'];

  if (status === 401) {
    return {
      type: 'auth',
      status,
      message: 'API anahtari gecersiz veya eksik. x-api-key ayarini kontrol edin.',
      retryAfterSec: null,
    };
  }

  if (status === 429) {
    return {
      type: 'rate_limit',
      status,
      message: 'Cok fazla istek gonderildi. Kisa sure sonra tekrar deneyin.',
      retryAfterSec: retryAfter ? Number(retryAfter) : null,
    };
  }

  if (status >= 500) {
    return {
      type: 'server',
      status,
      message: 'Sunucu tarafinda gecici bir hata olustu. Lutfen tekrar deneyin.',
      retryAfterSec: retryAfter ? Number(retryAfter) : null,
    };
  }

  return {
    type: 'network',
    status,
    message: error.response?.data?.detail || error.message || 'API baglantisinda hata olustu.',
    retryAfterSec: retryAfter ? Number(retryAfter) : null,
  };
};

const apiClient = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
    'x-api-key': API_KEY,
  },
  timeout: API_TIMEOUT_MS,
});

apiClient.interceptors.response.use(
  (response) => response,
  async error => {
    const config = error.config;
    const shouldRetry = !error.response || error.response.status >= 500 || error.code === 'ECONNABORTED';

    if (config && shouldRetry) {
      config.retryCount = config.retryCount || 0;

      if (config.retryCount < RETRY_COUNT) {
        config.retryCount += 1;
        const backoffMs = RETRY_DELAY_MS * config.retryCount;
        await sleep(backoffMs);
        return apiClient(config);
      }
    }

    return Promise.reject(error);
  }
);

export const EntropyService = {
  getConfig: () => ({
    apiUrl: API_URL,
    timeoutMs: API_TIMEOUT_MS,
    retryCount: RETRY_COUNT,
    hasApiKey: Boolean(API_KEY),
  }),

  parseError: (error) => humanizeApiError(error),

  /**
   * Entropi üret
   * Backend endpoint: POST /random/bytes
   */
  generateEntropy: async (bytes = 18) => {
    const startedAt = new Date().toISOString();
    const response = await apiClient.post('/random/bytes', {
        count: bytes
      });

    return {
        values: response.data.bytes,
        bytes: response.data.count,
        count: response.data.count,
        entropy_estimate: response.data.entropy_estimate,
        postprocessing: response.data.postprocessing,
        requestMeta: {
          startedAt,
          completedAt: new Date().toISOString(),
          status: response.status,
          requestId: response.headers['x-request-id'] || null,
        },
      };
  },

  /**
   * Sistem sağlığı kontrolü
   * Backend endpoint: GET /health
   */
  getHealth: async () => {
    const response = await apiClient.get('/health');
    return {
      ...response.data,
      _meta: {
        status: response.status,
        requestId: response.headers['x-request-id'] || null,
      },
    };
  },

  /**
   * İstatistikleri getir
    * Backend endpoint: yerel frontend istatistikleri kullanıyor, uyumluluk için /api/stats
   */
  getStats: async () => {
    try {
      const response = await apiClient.get('/api/stats');
      return response.data;
    } catch (error) {
      return {
        total_bytes: 0,
        generation_count: 0,
        reseed_count: 0,
        postprocessing: 'von_neumann',
        last_request_time: null,
        recent_errors: [
          {
            ts: new Date().toISOString(),
            event: 'stats_fetch_error',
            path: '/api/stats',
            status: error.response?.status || 0,
          },
        ],
      };
    }
  },
 
  /**
   * Kaotik sistemi yeniden tohumla
   * Backend endpoint: POST /chaos/reseed
   */
  reseedSystem: async (source = "os_entropy") => {
    const response = await apiClient.post('/chaos/reseed', { source });
    return response.data;
  }
};

// App.js'de kullanımı için uyumluluk
export default EntropyService;