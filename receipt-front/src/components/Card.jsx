export function Card({ children, className = '', title, subtitle }) {
  return (
    <div className={`
      bg-white 
      rounded-2xl shadow-md
      border border-gray-200/60
      backdrop-blur-sm
      p-8
      w-full
      transition-all duration-300
      hover:shadow-xl hover:border-gray-300
      ${className}
    `}>
      {title && (
        <div className="mb-6 pb-6 border-b-2 border-gray-200">
          <h2 className="text-2xl font-bold text-gray-900">{title}</h2>
          {subtitle && (
            <p className="text-sm text-gray-600 mt-2">{subtitle}</p>
          )}
        </div>
      )}
      {children}
    </div>
  );
}
