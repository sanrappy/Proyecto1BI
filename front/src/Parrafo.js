import { Link } from "react-router-dom";
import HomeIcon from './HomeIcon';
import './Parrafo.css';
import React, { useState, useEffect } from "react";

const Parrafo = () => {
  const [inputValue, setInputValue] = useState('');
  const [predClass, setPredClass] = useState('');
  const [resdata, setResData] = useState('');
  const [nerds, setNerds] = useState(false);
  const [isSubmitDisabled, setIsSubmitDisabled] = useState(true);

  useEffect(() => {
    console.log('predClass:', predClass);
}, [predClass]);

  async function handleClick(event) {
    event.preventDefault();
    try {
        const texto = {
            review: inputValue
        };

        console.log('Sending: ', texto);

        const response = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(texto),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log('Response data:', data);
        const predictedClass = data['predicted class'];
        setPredClass(predictedClass);
        setResData(data);
        
    } catch (error) {
        console.error('There was a problem with the fetch operation: ', error);
    }
}
  
function StarRating({ rating }) {
    return (
        <div className="flex justify-center text-4xl">
            {Array.from({ length: rating }, (_, i) => (
                <span key={i}>⭐</span>
            ))}
        </div>
    );
}
function fornerds() {
  setNerds(prevShowTextArea => !prevShowTextArea);
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
            <textarea
              className="rounded-md border h-20 bg-gray-100 w-full text-black p-3 justify-center"
              placeholder="Escriba una reseña aquí. Inserte al menos 20 palabras para el funcionamiento correcto del algoritmo."
              value={inputValue}
              onChange={e => {
                setInputValue(e.target.value);
                const wordCount = e.target.value.split(' ').filter(word => word).length;
                setIsSubmitDisabled(wordCount < 20);
              }}
            />
            <button
              className="inline-flex h-15 items-center justify-center rounded-md border border-blue-600 bg-blue-600 px-4 py-2 text-sm font-medium text-white shadow transition-colors hover:bg-blue-600/90 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-gray-950 disabled:pointer-events-none disabled:opacity-50 dark:border-blue-300 dark:bg-blue-600 dark:hover:bg-blue-600 dark:hover:text-gray-900 dark:focus-visible:ring-gray-300 w-full"
              onClick={handleClick}
              disabled={isSubmitDisabled}
            >
              Enviar Reseña
            </button>
            
            <StarRating rating={predClass}/>
            {predClass && (
                <div className="flex justify-center">
                  <button onClick={fornerds} className="inline-flex h-15 items-center justify-center rounded-md border border-black bg-white px-2 py-1 text-sm font-small text-black shadow transition-colors hover:bg-gray-100 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-gray-950 disabled:pointer-events-none disabled:opacity-50 w-1/3">
                      Press to see data for nerds
                  </button>
                </div>
            )}
            {nerds && (
  <textarea 
    className=" flex rounded-md border h-48 bg-gray-100 w-1/3 items-center justify-center text-black space-y-4 mx-auto" 
    placeholder="Aquí aparecerá el resultado..." 
    readOnly
    style={{ textAlign: 'center' }}
    value={`predicted class: ${resdata["predicted class"]}\npredicted probabilities for each class:\n${Object.entries(resdata["predicted probabilites for each class"]).map(([key, value]) => `${key}: ${value}`).join('\n')}`}
  />
)}
          </div>
        </div>
      </main>
    </>
  )
}

export default Parrafo;
