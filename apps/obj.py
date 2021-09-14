import numpy as np
import argparse
import os
import math

def make_rotate(rx, ry, rz):
    sinX = np.sin(rx)
    sinY = np.sin(ry)
    sinZ = np.sin(rz)

    cosX = np.cos(rx)
    cosY = np.cos(ry)
    cosZ = np.cos(rz)

    Rx = np.zeros((3,3))
    Rx[0, 0] = 1.0
    Rx[1, 1] = cosX
    Rx[1, 2] = -sinX
    Rx[2, 1] = sinX
    Rx[2, 2] = cosX

    Ry = np.zeros((3,3))
    Ry[0, 0] = cosY
    Ry[0, 2] = sinY
    Ry[1, 1] = 1.0
    Ry[2, 0] = -sinY
    Ry[2, 2] = cosY

    Rz = np.zeros((3,3))
    Rz[0, 0] = cosZ
    Rz[0, 1] = -sinZ
    Rz[1, 0] = sinZ
    Rz[1, 1] = cosZ
    Rz[2, 2] = 1.0

    R = np.matmul(np.matmul(Rz,Ry),Rx)
    return R

def save_obj_mesh(mesh_path, verts, faces, normdata, uvdata, facenorm, faceuv, mtlname, usemtl):
    file = open(mesh_path, 'w')
    
    file.write('mtllib %s \n' % mtlname)
    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    for n in normdata:
        file.write('vn %f %f %f\n' % (n[0], n[1], n[2]))
    for uv in uvdata:
        file.write('vt %f %f\n' % (uv[0], uv[1]))
    file.write('usemtl %s \n' % usemtl)

    if len(faceuv)!=0 and len(facenorm)!=0:
        for f, fu, fn in zip(faces, faceuv, facenorm):
            file.write('f %d/%d/%d %d/%d/%d %d/%d/%d\n' %(f[0], fu[0], fn[0], f[1], fu[1], fn[1], f[2], fu[2], fn[2]))
    elif len(faceuv)==0 and len(facenorm)==0:
        for f in faces:
            file.write('f %d %d %d\n' %(f[0], f[1], f[2]))
    elif len(faceuv)==0:
        for f, fn in zip(faces, facenorm):
            file.write('f %d/%d %d/%d %d/%d\n'% (f[0], fn[0], f[1], fn[1], f[2], fn[2]))
    elif len(facenorm)==0:
        for f, fuv in zip(faces, faceuv):
            file.write('f %d/%d %d/%d %d/%d\n'% (f[0], fuv[0], f[1], fuv[1], f[2], fuv[2]))
    

    file.close()

def load_obj_mesh(mesh_file, with_normal=False, with_texture=False):
    vertex_data = []
    norm_data = []
    uv_data = []

    face_data = []
    face_norm_data = []
    face_uv_data = []

    if isinstance(mesh_file, str):
        f = open(mesh_file, "r")
    else:
        f = mesh_file
    for line in f:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue

        if values[0] == 'v':
            v = list(map(float, values[1:4]))
            vertex_data.append(v)
        elif values[0] == 'vn':
            vn = list(map(float, values[1:4]))
            norm_data.append(vn)
        elif values[0] == 'vt':
            vt = list(map(float, values[1:3]))
            uv_data.append(vt)

        elif values[0] == 'f':
            # quad mesh
            if len(values) > 4:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)
                f = list(map(lambda x: int(x.split('/')[0]), [values[3], values[4], values[1]]))
                face_data.append(f)
            # tri mesh
            else:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)
            # deal with texture
            if len(values[1].split('/')) >= 2:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[1]), [values[3], values[4], values[1]]))
                    face_uv_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[1]) != 0:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
            # deal with normal
            if len(values[1].split('/')) == 3:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[2]), [values[3], values[4], values[1]]))
                    face_norm_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[2]) != 0:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)
                
        elif values[0] == 'mtllib':
            mtlname = values[1]
        elif values[0] == 'usemtl':
            usemtl = values[1]

    vertices = np.array(vertex_data)
    faces = np.array(face_data)
    normdata = np.array(norm_data)
    uvdata = np.array(uv_data)
    facenorm = np.array(face_norm_data)
    faceuv = np.array(face_uv_data)
    return vertices, faces, normdata, uvdata, facenorm, faceuv, mtlname, usemtl

def trans(inp, outp):
    vertices, faces, normdata, uvdata, facenorm, faceuv, mtlname, usemtl = load_obj_mesh(inp)
    vmin = vertices.min(0)
    vmax = vertices.max(0)
    #up_axis = 1 if (vmax-vmin).argmax() == 1 else 2
    vmed = np.median(vertices, 0)
    vmed[1] = 0.5*(vmax[1]+vmin[1])
    # alignment
    vmove = [vmed[0], vmin[1], vmed[2]]
    vertices = [i - vmove for i in vertices]
    # Rotate
    #R = make_rotate(0, math.radians(270), 0)
    #vertices = [np.matmul(R, i) for i in vertices]
    # Scale
    vertices = [i*100 for i in vertices]
    save_obj_mesh(outp, vertices, faces, normdata, uvdata, facenorm, faceuv, mtlname, usemtl)
    


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='/home/shunsuke/Downloads/rp_dennis_posed_004_OBJ')
    #parser.add_argument('-o', '--output', type=str, default='/home/shunsuke/Documents/hf_human')
    args = parser.parse_args()
    obj = ""
    for file in os.listdir(args.input):
        if file[-4:] == '.OBJ' or file[-4:] == '.obj':
            obj = file
    output = os.path.join(args.input, obj[:-4]+"_new.obj")
    file = os.path.join(args.input, obj)
    trans(file, output)
