import config from '../../config';
import axiosInstance from '../utils/axiosInstance';
import useMe from './useMe';

interface UseAuth {
  signIn: (email: string, password: string) => Promise<void>;
  signUp: (email: string, password: string) => Promise<void>;
  signOut: () => void;
}

const useAuth = (): UseAuth => {
  const { clearMe, updateMe } = useMe();

  const signIn = async (email: string, password: string): Promise<void> => {
    try {
      const res = await axiosInstance({
        baseURL: config.apiServerGraphQLURL,
        data: {
          query: `
            mutation SignIn($email: String!, $password: String!) {
              signIn(email: $email, password: $password) {
                jwtToken
              }
            }
          `,
          variables: {
            email,
            password,
          },
        },
      });

      if (res?.data?.data?.signIn?.jwtToken === null || res?.data?.data?.signIn?.jwtToken === '') {
        // eslint-disable-next-line no-console
        console.log('Failed to sign in.');
        return;
      }

      updateMe(res?.data?.data?.signIn);
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error('signIn', err);
    }
  };

  const signUp = async (email: string, password: string): Promise<void> => {
    try {
      const res = await axiosInstance({
        baseURL: config.apiServerGraphQLURL,
        data: {
          query: `
            mutation SignUp($email: String!, $password: String!) {
              signUp(email: $email, password: $password) {
                jwtToken
              }
            }
          `,
          variables: {
            email,
            password,
          },
        },
      });

      if (res?.data?.data?.signUp?.jwtToken === null || res?.data?.data?.signUp?.jwtToken === '') {
        // eslint-disable-next-line no-console
        console.log('Failed to sign up.');
        return;
      }

      updateMe(res?.data?.data?.signUp);
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error(err);
    }
  };

  const signOut = (): void => {
    clearMe();
  };

  return {
    signIn,
    signUp,
    signOut,
  };
};

export default useAuth;
