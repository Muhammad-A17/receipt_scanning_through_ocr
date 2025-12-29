export function Page({ children, className = '', centered = true }) {
  return (
    <div className={`
      min-h-screen 
      bg-white
      transition-colors duration-200
      ${centered ? 'flex items-center justify-center' : ''}
      p-4 sm:p-6 lg:p-8
      ${className}
    `}>
      {children}
    </div>
  );
}
