import type { AxiosResponse } from 'axios';
import config from '../../config';
import Me from '../types/Me';
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
      const res: AxiosResponse<{ data: { signIn: Me } }> = await axiosInstance({
        baseURL: config.graphqlServerGraphQLURL,
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
        console.log('Failed to sign in.');
        return;
      }
      updateMe(res?.data?.data?.signIn);
    } catch (err) {
      console.error('signIn', err);
    }
  };

  const signUp = async (email: string, password: string): Promise<void> => {
    try {
      const res: AxiosResponse<{ data: { signUp: Me } }> = await axiosInstance({
        baseURL: config.graphqlServerGraphQLURL,
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
        console.log('Failed to sign up.');
        return;
      }

      updateMe(res?.data?.data?.signUp);
    } catch (err) {
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
