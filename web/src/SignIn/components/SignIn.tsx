import { useMutation } from '@tanstack/react-query';
import React from 'react';
import type { SubmitHandler } from 'react-hook-form';
import { useForm } from 'react-hook-form';
import { useRouter } from '@tanstack/react-router';
import useAuth from '../../auth/hooks/useAuth';
import useMe from '../../auth/hooks/useMe';
import styles from './SignIn.module.css';

type FormValues = {
  email: string;
  password: string;
};

function SignIn() {
  const { signIn } = useAuth();
  const { me } = useMe();
  const router = useRouter();

  const mutation = useMutation({
    mutationFn: (data: FormValues) => {
      const { email, password } = data;
      return signIn(email, password);
    },
  });

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<FormValues>();

  const onSubmit: SubmitHandler<FormValues> = (data) => {
    mutation.mutate(data);
  };

  if (me != null) {
    router.history.push('/lab');
    return null;
  }

  return (
    <div className={styles.hmSignIn}>
      <div className={`container is-max-desktop ${styles.hmContainer}`}>
        <form onSubmit={handleSubmit(onSubmit)}>
          <div className="field">
            <div className="control">
              <input className="input" type="email" placeholder="Email" {...register('email', { required: true })} />
              {errors.email && <p className="help is-danger">Email is required.</p>}
            </div>
          </div>
          <div className="field">
            <div className="control">
              <input
                {...register('password', { required: true })}
                className="input"
                type="password"
                placeholder="Password"
              />
              {errors.password && <p className="help is-danger">Password is required.</p>}
            </div>
          </div>
          <div className="field">
            <div className="control">
              <button className="button is-primary is-fullwidth" type="submit">
                Sign In
              </button>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
}

export default SignIn;
