import { Link } from "react-router-dom";
import HomeIcon from './HomeIcon';
import './Parrafo.css';
import React, { useState } from "react";
import axios from 'axios';

const Parrafo = () => {
  const [inputValue, setInputValue] = useState('');
  const handleClick = async () => {
    const response = await axios.post('http://localhost:3000/parrafo', { "review": inputValue });
    console.log(response.data);
  }


  return (
    <>
      <div className="bg-blue-600">
        <nav className="container flex items-center h-14 px-4 md:px-6 mx-auto">
          <Link className="flex items-center space-x-2 text-sm font-medium text-white" to="/">
            <HomeIcon className="h-4 w-4" />
            <span>Inicio</span>
          </Link>
          <div className="ml-auto flex space-x-4 items-end">
            <Link className="font-medium text-white hover:underline dark:text-gray-50" to="/parrafo">
              Insertar Párrafo
            </Link>
            <Link className="font-medium text-white hover:underline dark:text-gray-50" to="/subarchivo">
              Subir un archivo
            </Link>
          </div>
        </nav>
      </div>
      <main>
        <div className="flex items-center justify-center space-y-4 mx-auto h-screen">
          <div className="w-2/3 space-y-4 mx-auto h-1/3">
            <input
              className="rounded-md border h-20 bg-gray-100 w-full text-black"
              placeholder="Escribe tu reseña"
              value={inputValue}
              onChange={e => setInputValue(e.target.value)}
            />
            <button
              className="inline-flex h-15 items-center justify-center rounded-md border border-blue-600 bg-blue-600 px-4 py-2 text-sm font-medium text-white shadow transition-colors hover:bg-blue-600/90 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-gray-950 disabled:pointer-events-none disabled:opacity-50 dark:border-blue-300 dark:bg-blue-600 dark:hover:bg-blue-600 dark:hover:text-gray-900 dark:focus-visible:ring-gray-300 w-full"
              onClick={handleClick}
            >
              Enviar Reseña
            </button>

            <textarea className="rounded-md border h-9 bg-gray-100 w-full h-64 items-center justify-center text-white space-y-4 mx-auto" placeholder="Aquí aparecerá el resultado..." readOnly />
          </div>
        </div>
      </main>
    </>
  )
}

export default Parrafo;
