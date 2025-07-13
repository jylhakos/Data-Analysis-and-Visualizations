import React, { useState, useEffect } from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Container,
  Grid,
  Card,
  CardContent,
  Box,
  Tab,
  Tabs,
  Alert,
  CircularProgress
} from '@mui/material';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import ThermostatIcon from '@mui/icons-material/Thermostat';
import OpacityIcon from '@mui/icons-material/Opacity';
import SpeedIcon from '@mui/icons-material/Speed';
import AirIcon from '@mui/icons-material/Air';

import WeatherChart from './components/WeatherChart';
import StationMap from './components/StationMap';
import RealTimeMetrics from './components/RealTimeMetrics';
import DataTable from './components/DataTable';
import { WeatherService } from './services/WeatherService';
import { WeatherMeasurement } from './types/WeatherTypes';

// Create Material-UI theme
const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#f5f5f5',
    },
  },
  typography: {
    h4: {
      fontWeight: 600,
    },
    h6: {
      fontWeight: 500,
    },
  },
});

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`weather-tabpanel-${index}`}
      aria-labelledby={`weather-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

function App() {
  const [tabValue, setTabValue] = useState(0);
  const [measurements, setMeasurements] = useState<WeatherMeasurement[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected' | 'connecting'>('connecting');

  const weatherService = new WeatherService();

  useEffect(() => {
    initializeData();
    setupRealTimeConnection();

    return () => {
      weatherService.disconnect();
    };
  }, []);

  const initializeData = async () => {
    try {
      setLoading(true);
      const data = await weatherService.getLatestMeasurements(24); // Last 24 hours
      setMeasurements(data);
      setError(null);
    } catch (err) {
      setError('Failed to load weather data');
      console.error('Error loading data:', err);
    } finally {
      setLoading(false);
    }
  };

  const setupRealTimeConnection = () => {
    setConnectionStatus('connecting');
    
    weatherService.connect();

    weatherService.onConnectionStatusChange((status) => {
      setConnectionStatus(status);
    });

    weatherService.onNewMeasurement((measurement) => {
      setMeasurements(prev => [measurement, ...prev.slice(0, 999)]); // Keep last 1000 measurements
    });

    weatherService.onError((errorMessage) => {
      setError(errorMessage);
    });
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const getLatestMeasurementByStation = () => {
    const stationMap = new Map<string, WeatherMeasurement>();
    
    measurements.forEach(measurement => {
      if (!stationMap.has(measurement.stationId) || 
          new Date(measurement.timestamp) > new Date(stationMap.get(measurement.stationId)!.timestamp)) {
        stationMap.set(measurement.stationId, measurement);
      }
    });

    return Array.from(stationMap.values());
  };

  const calculateAverages = () => {
    const latest = getLatestMeasurementByStation();
    
    if (latest.length === 0) {
      return { temperature: 0, humidity: 0, pressure: 0, windSpeed: 0 };
    }

    return {
      temperature: latest.reduce((sum, m) => sum + (m.temperature || 0), 0) / latest.length,
      humidity: latest.reduce((sum, m) => sum + (m.humidity || 0), 0) / latest.length,
      pressure: latest.reduce((sum, m) => sum + (m.pressure || 0), 0) / latest.length,
      windSpeed: latest.reduce((sum, m) => sum + (m.windSpeed || 0), 0) / latest.length,
    };
  };

  const averages = calculateAverages();

  if (loading) {
    return (
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <Box display="flex" justifyContent="center" alignItems="center" minHeight="100vh">
          <CircularProgress size={60} />
        </Box>
      </ThemeProvider>
    );
  }

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <AppBar position="static" elevation={2}>
        <Toolbar>
          <ThermostatIcon sx={{ mr: 2 }} />
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Weather ETL Dashboard
          </Typography>
          <Box
            sx={{
              width: 12,
              height: 12,
              borderRadius: '50%',
              backgroundColor: connectionStatus === 'connected' ? 'green' : 
                             connectionStatus === 'disconnected' ? 'red' : 'orange',
              mr: 1
            }}
          />
          <Typography variant="body2">
            {connectionStatus === 'connected' ? 'Live' : 
             connectionStatus === 'disconnected' ? 'Offline' : 'Connecting...'}
          </Typography>
        </Toolbar>
      </AppBar>

      <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
        {error && (
          <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        {/* Key Metrics Cards */}
        <Grid container spacing={3} sx={{ mb: 4 }}>
          <Grid item xs={12} sm={6} md={3}>
            <Card elevation={3}>
              <CardContent>
                <Box display="flex" alignItems="center">
                  <ThermostatIcon color="primary" sx={{ mr: 2, fontSize: 40 }} />
                  <Box>
                    <Typography color="textSecondary" gutterBottom variant="body2">
                      Avg Temperature
                    </Typography>
                    <Typography variant="h5" component="div">
                      {averages.temperature.toFixed(1)}°C
                    </Typography>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} sm={6} md={3}>
            <Card elevation={3}>
              <CardContent>
                <Box display="flex" alignItems="center">
                  <OpacityIcon color="primary" sx={{ mr: 2, fontSize: 40 }} />
                  <Box>
                    <Typography color="textSecondary" gutterBottom variant="body2">
                      Avg Humidity
                    </Typography>
                    <Typography variant="h5" component="div">
                      {averages.humidity.toFixed(1)}%
                    </Typography>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} sm={6} md={3}>
            <Card elevation={3}>
              <CardContent>
                <Box display="flex" alignItems="center">
                  <SpeedIcon color="primary" sx={{ mr: 2, fontSize: 40 }} />
                  <Box>
                    <Typography color="textSecondary" gutterBottom variant="body2">
                      Avg Pressure
                    </Typography>
                    <Typography variant="h5" component="div">
                      {averages.pressure.toFixed(0)} hPa
                    </Typography>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} sm={6} md={3}>
            <Card elevation={3}>
              <CardContent>
                <Box display="flex" alignItems="center">
                  <AirIcon color="primary" sx={{ mr: 2, fontSize: 40 }} />
                  <Box>
                    <Typography color="textSecondary" gutterBottom variant="body2">
                      Avg Wind Speed
                    </Typography>
                    <Typography variant="h5" component="div">
                      {averages.windSpeed.toFixed(1)} m/s
                    </Typography>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        {/* Tabs for different views */}
        <Card elevation={3}>
          <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tabs value={tabValue} onChange={handleTabChange} aria-label="weather dashboard tabs">
              <Tab label="Charts" />
              <Tab label="Station Map" />
              <Tab label="Real-Time Data" />
              <Tab label="Data Table" />
            </Tabs>
          </Box>

          <TabPanel value={tabValue} index={0}>
            <WeatherChart measurements={measurements} />
          </TabPanel>

          <TabPanel value={tabValue} index={1}>
            <StationMap measurements={getLatestMeasurementByStation()} />
          </TabPanel>

          <TabPanel value={tabValue} index={2}>
            <RealTimeMetrics 
              measurements={measurements.slice(0, 10)} 
              connectionStatus={connectionStatus}
            />
          </TabPanel>

          <TabPanel value={tabValue} index={3}>
            <DataTable measurements={measurements.slice(0, 100)} />
          </TabPanel>
        </Card>
      </Container>
    </ThemeProvider>
  );
}

export default App;
