export function Button({ children, onClick, className = '', color = 'primary', size = 'md', disabled = false, type = 'button' }) {
  const colorClasses = {
    primary: 'bg-blue-600 hover:bg-blue-700 active:bg-blue-800 text-white',
    secondary: 'bg-gray-600 hover:bg-gray-700 active:bg-gray-800 text-white',
    success: 'bg-green-600 hover:bg-green-700 active:bg-green-800 text-white',
    danger: 'bg-red-600 hover:bg-red-700 active:bg-red-800 text-white',
    outline: 'bg-transparent border-2 border-gray-600 hover:border-gray-700 text-gray-700 hover:bg-gray-50 font-semibold'
  };

  const sizeClasses = {
    sm: 'px-4 py-2 text-sm',
    md: 'px-6 py-3 text-base',
    lg: 'px-8 py-4 text-lg'
  };

  return (
    <button 
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`
        ${colorClasses[color] || colorClasses.primary}
        ${sizeClasses[size] || sizeClasses.md}
        font-semibold rounded-lg 
        transition-all duration-200 
        disabled:opacity-50 disabled:cursor-not-allowed
        focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500
        ${className}
      `}
    >
      {children}
    </button>
  );
}
