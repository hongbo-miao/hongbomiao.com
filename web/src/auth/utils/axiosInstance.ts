import axios from 'axios';

const axiosInstance = axios.create({
  headers: { 'Content-Type': 'application/json' },
  method: 'POST',
});

export default axiosInstance;
