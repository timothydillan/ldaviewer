import * as React from 'react';
import Backdrop from '@mui/material/Backdrop';
import { Typography } from '@mui/material';
import CircularProgress from '@mui/material/CircularProgress';
import Box from '@mui/material/Box';
import Grid from '@mui/material/Grid';

export default function BackdropProgress({ open, message }) {
    return (
        <Backdrop
            sx={{ color: '#fff', zIndex: (theme) => theme.zIndex.drawer + 1 }}
            open={open}
        >
            <Grid container marginTop={1} direction="column" alignItems="center" justifyContent="center">

                <Typography align="center" marginLeft={10} marginRight={10}>
                    {message}
                </Typography>
                <CircularProgress />
            </Grid>
        </Backdrop>
    );
}