import React from 'react';
import Paper from '@mui/material/Paper';
import InputBase from '@mui/material/InputBase';
import Divider from '@mui/material/Divider';
import IconButton from '@mui/material/IconButton';
import MenuIcon from '@mui/icons-material/Menu';
import SearchIcon from '@mui/icons-material/Search';
import DirectionsIcon from '@mui/icons-material/Directions';

export default function SearchBar({ placeholder, onSearchChange, onButtonClick, onMenuClick, onSubmit }) {
    return (
        <Paper
            component="form"
            sx={{ p: '2px 4px', display: 'flex', alignItems: 'center', width: 600 }}
            onSubmit={onSubmit}
        >
            <IconButton sx={{ p: '10px' }} aria-label="menu" onClick={() => { onMenuClick() }} >
                <MenuIcon />
            </IconButton>
            <InputBase
                sx={{ ml: 1, flex: 1 }}
                placeholder={placeholder}
                inputProps={{ 'aria-label': placeholder }}
                onChange={(e) => {
                    onSearchChange(e.target.value)
                }}
            />
            <Divider sx={{ height: 28, m: 0.5 }} orientation="vertical" />
            <IconButton color="primary" type="button" sx={{ p: '10px' }} aria-label="search" onClick={() => {
                onButtonClick()
            }}>
                <SearchIcon />
            </IconButton>
        </Paper>
    );
}
