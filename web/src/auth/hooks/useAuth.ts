/* eslint-disable no-console */

import axiosInstance from '../utils/axiosInstance';
import useMe from './useMe';

interface UseAuth {
  signIn: (email: string, password: string) => Promise<void>;
  signUp: (email: string, password: string) => Promise<void>;
  signOut: () => void;
}

const useAuth = (): UseAuth => {
  const { clearMe, updateMe } = useMe();

  async function signIn(email: string, password: string): Promise<void> {
    try {
      const res = await axiosInstance({
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
  }
  async function signUp(email: string, password: string): Promise<void> {
    try {
      const res = await axiosInstance({
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
  }

  function signOut(): void {
    clearMe();
  }

  return {
    signIn,
    signUp,
    signOut,
  };
};

export default useAuth;
