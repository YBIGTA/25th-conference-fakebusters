import type { Metadata } from "next";
import { Provider } from "@/components/ui/provider"
import Header from '../components/Header'
import "./globals.css";


import { Roboto_Condensed } from 'next/font/google'

const robotoCondensed = Roboto_Condensed({
  weight: ['300', '400', '700'],  
  style: ['normal', 'italic'],    
  subsets: ['latin'],             
  variable: '--font-roboto-condensed',
})

export const metadata: Metadata = {
  title: "Deepfake Detector",
  description: "Detects possibility of forging for images.",
  icons: {
    icon: '/icon.png',
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
      <html suppressHydrationWarning lang="en" className={robotoCondensed.variable}>
        <body>
          <Header />
          <Provider>{children}</Provider>
        </body>
      </html>
  )
}
