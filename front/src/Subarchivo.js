import { Link } from "react-router-dom";
import HomeIcon from './HomeIcon';
import './Parrafo.css';
import React, { createRef, useState } from "react";

const Subarchivo = () => {
    const file = createRef();

    async function handleSubmit(event) {
    event.preventDefault();
    try {

        if (!file.current.files[0]) {
            console.log('No file selected');
            return;
        }else{
            console.log("file ok!");
        }

        const formData = {
            file_path: file.current.files[0].name
        };

        console.log('Sending: ', formData);

        const response = await fetch('http://localhost:8000/retrain_model_with_new_datafile', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        else {
            alert("Archivo subido");
        }

        const data = await response.json();
        console.log(data);
    } catch (error) {
        console.error('There was a problem with the fetch operation: ', error);
    }
}
    async function handleResults(event) {
        event.preventDefault();
        try {
            const response = await fetch('http://localhost:8000/test_model', {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json'
                },
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            console.log(data);
        } catch (error) {
            console.error('There was a problem with the fetch operation: ', error);
        }
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
                {/*<div className="flex items-center justify-center space-y-4 mx-auto h-screen">
                    <div className="w-2/3 space-y-4 mx-auto h-1/3">
                        <form onSubmit={handleSubmit}>
                            <div class="flex items-center justify-center w-full">
                                <label for="dropzone-file" class="flex flex-col items-center justify-center w-full h-24 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 dark:hover:bg-bray-800 hover:bg-gray-100">
                                    <div class="flex flex-col items-center justify-center pt-5 pb-5">
                                        <svg class="w-8 h-8 mb-4 text-gray-500" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 20 16">
                                            <path stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 13h3a3 3 0 0 0 0-6h-.025A5.56 5.56 0 0 0 16 6.5 5.5 5.5 0 0 0 5.207 5.021C5.137 5.017 5.071 5 5 5a4 4 0 0 0 0 8h2.167M10 15V6m0 0L8 8m2-2 2 2" />
                                        </svg>
                                        <p class=" text-sm text-gray-500 dark:text-gray-400"><span class="font-semibold">Presione para subir un archivo .csv.</span></p>
                                    </div>
                                    <input id="dropzone-file" type="file" name="flupload" accept=".csv" ref={file}/>
                                </label>
                            </div>
                            <button
                                className="inline-flex h-15 items-center justify-center rounded-md border border-blue-600 bg-blue-600 px-4 py-2 text-sm font-medium text-white shadow transition-colors hover:bg-blue-600/90 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-gray-950 disabled:pointer-events-none disabled:opacity-50 dark:border-blue-300 dark:bg-blue-600 dark:hover:bg-blue-600 dark:hover:text-gray-900 dark:focus-visible:ring-gray-300 w-full"
                                type="submit"
                                value="Submit"
                            >
                                Analizar archivo de reseñas
                            </button>
                        </form>
                    </div>
            </div>*/}
            <form onSubmit={handleSubmit}>
            <input type="file" ref={file} />
            <button type="submit">Submit</button>
            
        </form>
        <button onClick={handleResults}>Mostrar resultados</button>
            </main>
        </>
    );
}

export default Subarchivo;