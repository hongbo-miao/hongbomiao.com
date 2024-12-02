import { createSlice } from '@reduxjs/toolkit';
import type { PayloadAction } from '@reduxjs/toolkit';

interface MeState {
  name: string;
  bio: string;
}

const initialState: MeState = {
  name: 'Hongbo Miao',
  bio: 'Making magic happen',
};

const meSlice = createSlice({
  name: 'me',
  initialState,
  reducers: {
    queryMeSucceed(state, action: PayloadAction<{ res: { response: { data: { me: Partial<MeState> } } } }>) {
      const { res } = action.payload;
      return {
        ...state,
        ...(res.response.data.me || {}),
      };
    },
  },
});

export const { queryMeSucceed } = meSlice.actions;
export default meSlice.reducer;
