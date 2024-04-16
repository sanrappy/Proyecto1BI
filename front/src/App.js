import logo from './logo.svg';
import './App.css';
import { Link } from "react-router-dom";

function App() {
  return (
    <>
      <div className="bg-blue-600">
        <nav className="container flex items-center h-14 px-4 md:px-6 mx-auto">
          <Link className="flex items-center space-x-2 text-sm font-medium text-white">
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
        <div className="container flex flex-col items-center justify-center min-h-[calc(100vh_-_theme(spacing.24))_] py-6 px-4 md:px-6 gap-4 text-center mx-auto">
          <div className="space-y-2">
            <h1 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl lg:text-6xl/none text-gray-900 dark:text-gray-900">
              Etapa 2 del Proyecto 1 de Inteligencia de Negocios
            </h1>
            <p className="max-w-[700px] text-gray-500 md:text-xl/relaxed lg:text-base/relaxed xl:text-xl/relaxed dark:text-gray-800 mx-auto">
              Somos un grupo conformado por Daniel Santiago Rappy, Juan Andrés Jaramillo y Nicole Murillo
            </p>
          </div>
        </div>
      </main>
    </>
  )
}

function HomeIcon(props) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
      <polyline points="9 22 9 12 15 12 15 22" />
    </svg>
  )
}

export default App;
