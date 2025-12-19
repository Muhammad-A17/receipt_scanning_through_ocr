import { theme } from '../theme.js';

export function Page({ children, className = '' }) {
  return (
    <div className={`${theme.layouts.page} ${className}`}>
      {children}
    </div>
  );
}