import axios from 'axios';
import config from '../../config';

const axiosInstance = axios.create({
  baseURL: config.graphQLURL,
  headers: { 'Content-Type': 'application/json' },
  method: 'POST',
});

export default axiosInstance;
