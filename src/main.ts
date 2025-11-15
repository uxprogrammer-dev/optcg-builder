import { Logger, ValidationPipe } from '@nestjs/common';
import { NestFactory } from '@nestjs/core';
import { ConfigService } from '@nestjs/config';
import { AppModule } from './app.module';

async function bootstrap() {
  const app = await NestFactory.create(AppModule);

  app.setGlobalPrefix('api');
  app.useGlobalPipes(
    new ValidationPipe({
      whitelist: true,
      forbidNonWhitelisted: true,
      transform: true,
      transformOptions: { enableImplicitConversion: true },
    }),
  );

  const configService = app.get(ConfigService);
  const frontendOrigin = configService.get<string>('frontend.origin') ?? 'http://localhost:3001';
  const nodeEnv = process.env.NODE_ENV ?? 'development';
  const isDevelopment = nodeEnv === 'development';
  
  // Support multiple origins (comma-separated) or single origin
  const allowedOrigins = frontendOrigin
    .split(',')
    .map((origin) => origin.trim())
    .filter(Boolean);

  // Function to check if origin is allowed (returns origin if allowed, false otherwise)
  const originChecker = (origin: string | undefined): string | boolean => {
    // Allow requests with no origin (like mobile apps or curl requests)
    if (!origin) {
      return true;
    }

    // In development, be more permissive to handle ngrok and other tunneling services
    if (isDevelopment) {
      // Allow all localhost origins
      if (origin.startsWith('http://localhost:') || origin.startsWith('https://localhost:')) {
        return origin;
      }
      
      // Allow all ngrok origins
      if (origin.includes('.ngrok-free.app') || origin.includes('.ngrok.io') || origin.includes('.ngrok.app')) {
        return origin;
      }
      
      // Allow any local network IPs (for development)
      if (origin.match(/^https?:\/\/(127\.0\.0\.1|192\.168\.|10\.|172\.(1[6-9]|2[0-9]|3[01])\.)/)) {
        return origin;
      }
    }

    // Check if origin is in allowed list
    if (allowedOrigins.includes(origin)) {
      return origin;
    }

    // Deny by default
    return false;
  };

  // In development, allow all origins to handle ngrok and other tunneling services
  // This is necessary because ngrok intercepts OPTIONS requests and shows a warning page
  const corsConfig = isDevelopment
    ? {
        origin: true, // Allow all origins in development
        credentials: true,
        methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'],
        allowedHeaders: [
          'Content-Type',
          'Authorization',
          'Accept',
          'ngrok-skip-browser-warning',
          'X-Requested-With',
          'Access-Control-Request-Method',
          'Access-Control-Request-Headers',
          'Origin',
        ],
        exposedHeaders: ['Content-Length', 'Content-Type'],
        preflightContinue: false,
        optionsSuccessStatus: 204,
        maxAge: 86400, // 24 hours
      }
    : {
        origin: originChecker,
        credentials: true,
        methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'],
        allowedHeaders: [
          'Content-Type',
          'Authorization',
          'Accept',
          'ngrok-skip-browser-warning',
          'X-Requested-With',
          'Access-Control-Request-Method',
          'Access-Control-Request-Headers',
        ],
        exposedHeaders: ['Content-Length', 'Content-Type'],
        preflightContinue: false,
        optionsSuccessStatus: 204,
        maxAge: 86400,
      };

  app.enableCors(corsConfig);
  
  if (isDevelopment) {
    Logger.log('CORS enabled for development - allowing all origins', 'Bootstrap');
  }

  const port = configService.get<number>('PORT') ?? 3000;

  await app.listen(port);
  Logger.log(`Server listening on http://localhost:${port}`, 'Bootstrap');
}

bootstrap();

