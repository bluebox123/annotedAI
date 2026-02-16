import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'multipart/form-data',
  },
});

export const uploadPDFs = async (files) => {
  const formData = new FormData();
  files.forEach(file => {
    formData.append('files', file);
  });
  
  const response = await api.post('/api/upload', formData);
  return response.data;
};

export const askQuestion = async (question, highlightEngine = 'keyword', restrictContext = true, history = []) => {
  const formData = new FormData();
  formData.append('question', question);
  formData.append('highlight_engine', highlightEngine);
  formData.append('restrict_context', restrictContext);
  formData.append('history', JSON.stringify(history || []));
  
  const response = await api.post('/api/ask', formData);
  return response.data;
};

export const getPreviewUrl = (previewId) => {
  return `${API_BASE_URL}/api/preview/${previewId}`;
};

export const listFiles = async () => {
  const response = await api.get('/api/files');
  return response.data;
};

export const clearFiles = async () => {
  const response = await api.delete('/api/files');
  return response.data;
};
