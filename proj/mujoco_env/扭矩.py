import math

def calcu(current,resistence_25c,temp_celsius=25.0,kv_nominal=5200,load_factor=1.0):
    resistance_actual=resistence_25c*(1+0.00393*(temp_celsius-25))

    kv_compensated=kv_nominal*(1-0.0015*(temp_celsius-25))

    efficiency_map={
            0.5:0.72,0.8:0.78,1.0:0.81,1.2:0.80,1.4:0.76}
    
    currents=sorted(efficiency_map.keys())
    eff=efficiency_map[currents[0]]
    for i in range(len(currents)-1):
        if currents[i] <= current < currents[i+1]:
            eff=(efficiency_map[currents[i]]*(currents[i+1]-current)
                 +efficiency_map[currents[i+1]]*(current-currents[i]))/(currents[i+1]-currents[i])
            break
    kt=60/(2*math.pi*kv_compensated)*load_factor

    torque=kt*current*eff        
    return round(torque,5)

result=calcu(
    current=1.2,
    resistence_25c=0.092,
    temp_celsius=45.0,
    kv_nominal=5200,
    load_factor=0.92)
print(f"45 摄氏度下,1.2A电流产生的扭矩约为{result}Nm")
