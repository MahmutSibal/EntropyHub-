import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://127.0.0.1:8000';

const apiClient = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
    'x-api-key': 'teknofest-local-dev-key',
  },
  timeout: 15000,
});

apiClient.interceptors.response.use(
  response => response,
  async error => {
    const config = error.config;
    console.error("API Connection Error:", error.response ? error.response.data : error.message);
    
    // Otomatik Retry (Yeniden Deneme) Mekanizması
    if (config && (!config.__isRetryRequest) && (!error.response || error.response.status >= 500)) {
      config.retryCount = config.retryCount || 0;
      const maxRetries = 2; // Maksimum deneme sayısı
      
      if (config.retryCount < maxRetries) {
        config.retryCount += 1;
        config.__isRetryRequest = true;
        console.log(`Connection lost, retrying for the ${config.retryCount}. time...`);
        await new Promise(resolve => setTimeout(resolve, 1000)); // 1 saniye bekle
        return apiClient(config);
      }
    }

    return Promise.reject(error);
  }
);

export const EntropyService = {
  /**
   * Entropi üret
   * Backend endpoint: POST /random/bytes
   */
  generateEntropy: async (bytes = 18) => {
    try {
      const response = await apiClient.post('/random/bytes', {
        count: bytes
      });
      
      console.log('API Response:', response.data);

      return {
        values: response.data.bytes,
        bytes: response.data.count,
        count: response.data.count,
        entropy_estimate: response.data.entropy_estimate,
        postprocessing: response.data.postprocessing,
      };
      
    } catch (error) {
      console.error('Entropy generation error:', error);
      throw error;
    }
  },

  /**
   * Sistem sağlığı kontrolü
   * Backend endpoint: GET /health
   */
  getHealth: async () => {
    try {
      const response = await apiClient.get('/health');
      return response.data;
    } catch (error) {
      console.error('Health check error:', error);
      throw error;
    }
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
      console.error('Stats error:', error);
      // Hata durumunda varsayılan değerler döndür
      return { total_bytes: 0, generation_count: 0 };
    }
  },
 
  /**
   * Kaotik sistemi yeniden tohumla
   * Backend endpoint: POST /chaos/reseed
   */
  reseedSystem: async (source = "os_entropy") => {
    try {
      const response = await apiClient.post('/chaos/reseed', { source });
      return response.data;
    } catch (error) {
      console.error('Reseed error:', error); 
      throw error;
    }
  }
};

// App.js'de kullanımı için uyumluluk
export default EntropyService;