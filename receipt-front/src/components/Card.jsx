export function Card({ children, className = '' }) {
  return (
    <div className={`bg-gray-600 rounded-lg shadow-md p-6 max-w-md w-full ${className}`}>
      {children}
    </div>
  );
}