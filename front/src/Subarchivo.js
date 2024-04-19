import { Link } from "react-router-dom";
import HomeIcon from './HomeIcon';
import './Parrafo.css';
import React, { createRef, useState } from "react";

const Subarchivo = () => {
    const file = createRef();
    const [fileName, setFileName] = useState('Seleccione un archivo');
    const [mensaje, setMensaje] = useState('');
    const [resultados, setResultados] = useState(null);
    const [isLoading, setIsLoading] = useState(false);

    const handleFileChange = (e) => {
        setFileName(e.target.files[0].name);
    }

    async function handleSubmit(event) {
        event.preventDefault();
        try {

            if (!file.current.files[0]) {
                console.log('No file selected');
                return;
            } else {
                console.log("file ok!");
            }

            setIsLoading(true);

            const formData = new FormData();
            formData.append('file', file.current.files[0]);

            console.log('Sending: ', formData);

            const response = await fetch('http://localhost:8000/retrain_model_with_new_datafile', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            console.log(data);
            setIsLoading(false);
            setMensaje(data['message']);
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
            setResultados(data);

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
                <div className="flex items-center justify-center space-y-4 mx-auto h-screen">
                    <div className="w-2/3 space-y-4 mx-auto h-1/3">
                        <form onSubmit={handleSubmit}>
                            <div className="flex items-center justify-center w-full">
                                <label className="w-auto flex flex-col items-center px-4 py-6 bg-white text-blue rounded-lg shadow-lg tracking-wide uppercase border border-blue cursor-pointer hover:bg-blue hover:text-blue-900">
                                    <svg className="w-8 h-8" fill="currentColor" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
                                        <path d="M10 4a2 2 0 00-2 2v4a2 2 0 002 2 2 2 0 002-2V6a2 2 0 00-2-2zm0 12a6 6 0 100-12 6 6 0 000 12z" />
                                    </svg>
                                    <span className="mt-2 text-base leading-normal">{fileName}</span>
                                    <input type='file' className="hidden" ref={file} onChange={handleFileChange} />
                                </label>
                                <button className="ml-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700" type="submit">Enviar</button>
                            </div>
                        </form>
                        {isLoading ? (
                            <div className="flex text-xs justify-center mx-auto font-mono">Loading...</div>
                        ) : (
                            mensaje && (
                                <div className="flex text-xs justify-center mx-auto font-mono">{mensaje}</div>
                            )
                        )}
                        <button className="ml-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 w-full" onClick={handleResults}>Mostrar resultados</button>
                        {resultados && Object.entries(resultados).map(([key, value]) => (
                            <div className="flex text-[10px] justify-center leading-[10px] font-mono mx-auto">{key}: {value}</div>
                        ))}
                    </div>
                </div>

            </main >
        </>
    );
}

export default Subarchivo;