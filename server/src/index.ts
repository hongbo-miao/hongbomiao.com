import app from './app';
import createHTTP2Server from './app/utils/createHTTP2Server';
import isProd from './app/utils/isProd';
import Config from './config';
import initSentry from './log/utils/initSentry';
import printStatus from './log/utils/printStatus';

initSentry();

if (isProd) {
  app.listen(Config.port, printStatus);
} else {
  const http2Server = createHTTP2Server(app);
  http2Server.listen(Config.port, printStatus);
}
