import { useState, useEffect } from 'react'
import { GetServerSideProps } from 'next'
import Head from 'next/head'
import WeatherMap from '../components/WeatherMap'
import StationDashboard from '../components/StationDashboard'
import RealTimeChart from '../components/RealTimeChart'
import AlertPanel from '../components/AlertPanel'
import SystemHealth from '../components/SystemHealth'
import { WeatherData, StationStats, Alert, HealthStatus } from '../types/weather'
import { fetchWeatherData, fetchStationStats, fetchAlerts, fetchSystemHealth } from '../lib/api'

interface HomeProps {
  initialData: WeatherData[]
  initialStats: StationStats[]
  initialAlerts: Alert[]
  initialHealth: HealthStatus[]
}

export default function Home({ 
  initialData, 
  initialStats, 
  initialAlerts, 
  initialHealth 
}: HomeProps) {
  const [weatherData, setWeatherData] = useState<WeatherData[]>(initialData)
  const [stationStats, setStationStats] = useState<StationStats[]>(initialStats)
  const [alerts, setAlerts] = useState<Alert[]>(initialAlerts)
  const [systemHealth, setSystemHealth] = useState<HealthStatus[]>(initialHealth)
  const [selectedStation, setSelectedStation] = useState<string | null>(null)
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date())

  // Real-time data updates
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        // Fetch latest data
        const [newData, newStats, newAlerts, newHealth] = await Promise.all([
          fetchWeatherData({ limit: 50 }),
          Promise.all(
            stationStats.map(stat => fetchStationStats(stat.station_id))
          ),
          fetchAlerts(),
          fetchSystemHealth()
        ])

        setWeatherData(newData.data)
        setStationStats(newStats)
        setAlerts(newAlerts)
        setSystemHealth(newHealth)
        setLastUpdate(new Date())
      } catch (error) {
        console.error('Error updating data:', error)
      }
    }, 60000) // Update every minute

    return () => clearInterval(interval)
  }, [stationStats])

  // Filter data for selected station
  const filteredData = selectedStation 
    ? weatherData.filter(data => data.station_id === selectedStation)
    : weatherData

  return (
    <>
      <Head>
        <title>Weather ETL Dashboard</title>
        <meta name="description" content="Real-time weather data visualization dashboard" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div className="min-h-screen bg-gray-50">
        {/* Header */}
        <header className="bg-white shadow-sm border-b">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between items-center py-4">
              <div>
                <h1 className="text-2xl font-bold text-gray-900">
                  Weather ETL Dashboard
                </h1>
                <p className="text-sm text-gray-500">
                  Last updated: {lastUpdate.toLocaleTimeString()}
                </p>
              </div>
              <div className="flex items-center space-x-4">
                <SystemHealth data={systemHealth} />
              </div>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {/* Alerts */}
          {alerts.length > 0 && (
            <div className="mb-8">
              <AlertPanel alerts={alerts} />
            </div>
          )}

          {/* Top Section: Map and Station Selector */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
            <div className="lg:col-span-2">
              <div className="bg-white rounded-lg shadow p-6">
                <h2 className="text-lg font-semibold mb-4">Weather Stations Map</h2>
                <WeatherMap 
                  stations={stationStats}
                  onStationSelect={setSelectedStation}
                  selectedStation={selectedStation}
                />
              </div>
            </div>
            
            <div className="space-y-6">
              <div className="bg-white rounded-lg shadow p-6">
                <h2 className="text-lg font-semibold mb-4">Station Overview</h2>
                <div className="space-y-4">
                  {stationStats.map(station => (
                    <div 
                      key={station.station_id}
                      className={`p-4 rounded-lg border-2 cursor-pointer transition-colors ${
                        selectedStation === station.station_id
                          ? 'border-blue-500 bg-blue-50'
                          : 'border-gray-200 hover:border-gray-300'
                      }`}
                      onClick={() => setSelectedStation(
                        selectedStation === station.station_id ? null : station.station_id
                      )}
                    >
                      <div className="flex justify-between items-center">
                        <div>
                          <h3 className="font-medium">Station {station.station_id}</h3>
                          <p className="text-sm text-gray-500">
                            {station.data_points_24h} data points (24h)
                          </p>
                        </div>
                        <div className="text-right">
                          <p className="text-lg font-semibold">
                            {station.latest_reading?.temperature.toFixed(1)}°C
                          </p>
                          <p className="text-sm text-gray-500">
                            {station.latest_reading?.humidity.toFixed(0)}% humidity
                          </p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Charts Section */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-lg font-semibold mb-4">
                Real-time Temperature
                {selectedStation && ` - Station ${selectedStation}`}
              </h2>
              <RealTimeChart 
                data={filteredData}
                dataKey="temperature"
                unit="°C"
                color="#ef4444"
              />
            </div>
            
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-lg font-semibold mb-4">
                Humidity Levels
                {selectedStation && ` - Station ${selectedStation}`}
              </h2>
              <RealTimeChart 
                data={filteredData}
                dataKey="humidity"
                unit="%"
                color="#3b82f6"
              />
            </div>
          </div>

          {/* Detailed Station Dashboard */}
          {selectedStation && (
            <div className="mb-8">
              <StationDashboard 
                stationId={selectedStation}
                data={filteredData}
                stats={stationStats.find(s => s.station_id === selectedStation)}
              />
            </div>
          )}

          {/* Data Table */}
          <div className="bg-white rounded-lg shadow">
            <div className="px-6 py-4 border-b border-gray-200">
              <h2 className="text-lg font-semibold">Latest Weather Data</h2>
            </div>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Station
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Temperature
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Humidity
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Pressure
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Wind
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Time
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {filteredData.slice(0, 20).map((data, index) => (
                    <tr key={index} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                        {data.station_id}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {data.temperature.toFixed(1)}°C
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {data.humidity.toFixed(0)}%
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {data.pressure.toFixed(1)} hPa
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {data.wind_speed.toFixed(1)} km/h
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {new Date(data.timestamp).toLocaleString()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </main>
      </div>
    </>
  )
}

export const getServerSideProps: GetServerSideProps = async (context) => {
  try {
    // Fetch initial data on server side
    const [weatherResponse, alertsResponse, healthResponse] = await Promise.all([
      fetchWeatherData({ limit: 100 }),
      fetchAlerts(),
      fetchSystemHealth()
    ])

    // Get unique station IDs and fetch their stats
    const stationIds = [...new Set(weatherResponse.data.map(d => d.station_id))]
    const stationStats = await Promise.all(
      stationIds.map(id => fetchStationStats(id))
    )

    return {
      props: {
        initialData: weatherResponse.data,
        initialStats: stationStats,
        initialAlerts: alertsResponse,
        initialHealth: healthResponse,
      },
    }
  } catch (error) {
    console.error('Error fetching initial data:', error)
    
    return {
      props: {
        initialData: [],
        initialStats: [],
        initialAlerts: [],
        initialHealth: [],
      },
    }
  }
}
