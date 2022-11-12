import logger from '../../log/utils/logger';
import createCircuitBreaker from './createCircuitBreaker';

describe('createCircuitBreaker', () => {
  const loggerInfoSpy = jest.spyOn(logger, 'info').mockImplementation(() => null);
  const asyncFunc = jest.fn(async () => 'World');
  const breaker = createCircuitBreaker(asyncFunc);

  beforeEach(() => {
    jest.clearAllMocks();
    breaker.close();
  });

  afterAll(() => {
    breaker.shutdown();
  });

  test('call the input async function', async () => {
    expect.assertions(2);
    const input = 'Hello';
    const res = await breaker.fire(input);
    expect(asyncFunc).toHaveBeenCalledWith(input);
    expect(res).toBe('World');
  });

  test('log when the circuit opens', () => {
    breaker.open();
    expect(loggerInfoSpy.mock.calls).toMatchInlineSnapshot(`
      [
        [
          {
            "breakerName": "mockConstructor",
            "eventName": "open",
          },
          "createCircuitBreaker",
        ],
      ]
    `);
  });

  test('log when the circuit closes', () => {
    breaker.open();
    jest.clearAllMocks();
    breaker.close();
    expect(loggerInfoSpy.mock.calls).toMatchInlineSnapshot(`
      [
        [
          {
            "breakerName": "mockConstructor",
            "eventName": "close",
          },
          "createCircuitBreaker",
        ],
      ]
    `);
  });
});
