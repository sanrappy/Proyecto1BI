import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import Califvar from './Califvar';
import Parrafo from './Parrafo';
import Subarchivo from './Subarchivo';
import reportWebVitals from './reportWebVitals';
import { BrowserRouter } from 'react-router-dom';
import { Route, Routes } from 'react-router-dom';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route exact path="/" element={<App />} />
        <Route path="/parrafo" element={<Parrafo />} />
        <Route path="/subarchivo" element={<Subarchivo />} />
        <Route path="/califvar" element={<Califvar />} />
      </Routes>
    </BrowserRouter>
  </React.StrictMode>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
