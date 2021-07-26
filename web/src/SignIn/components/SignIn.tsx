/* eslint-disable react/jsx-props-no-spreading */

import React from 'react';
import { useForm } from 'react-hook-form';
import { useMutation } from 'react-query';
import useAuth from '../../auth/hooks/useAuth';
import styles from './SignIn.module.css';

const SignIn: React.VFC = () => {
  const { signIn } = useAuth();

  const mutation = useMutation((data: { email: string; password: string }) => {
    const { email, password } = data;
    return signIn(email, password);
  });

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm();

  // eslint-disable-next-line @typescript-eslint/ban-ts-comment
  // @ts-ignore
  const onSubmit = (data) => {
    mutation.mutate(data);
  };

  return (
    <div className={styles.hmSignIn}>
      <div className={`container is-max-desktop ${styles.hmContainer}`}>
        <form onSubmit={handleSubmit(onSubmit)}>
          <div className="field">
            <p className="control">
              <input className="input" type="email" placeholder="Email" {...register('email', { required: true })} />
              {errors.email && <p className="help is-danger">Email is required.</p>}
            </p>
          </div>
          <div className="field">
            <p className="control">
              <input
                {...register('password', { required: true })}
                className="input"
                type="password"
                placeholder="Password"
              />
              {errors.password && <p className="help is-danger">Password is required.</p>}
            </p>
          </div>
          <div className="field">
            <div className="control">
              <button className="button is-link" type="submit">
                Sign In
              </button>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
};

export default SignIn;
