import sys
import mdtraj as md
from numpy import linalg as LA
import numpy as np
import math


def Cal_CofM(traj_,atom_ndx):
    '''
    compute center of mass for molecules
    atom_ndx: atom index constituting the target molecule
    '''
    top_ = traj_.topology
    masses = np.array([top_.atom(a).element.mass for a in atom_ndx])
    masses /= masses.sum()
    return np.einsum('ijk,k->ij',np.swapaxes(traj_.xyz[:,atom_ndx,:].astype('float64'),1,2), masses)

def Cal_dist(a,b):
    #ab_dist = np.sum((a - b)**2,axis=2)**(1/2.)
    if a.ndim == 3:
        return np.sum((a - b)**2,axis=2)**(1/2.)
    elif a.ndim == 2:
        return np.sum((a - b)**2,axis=1)**(1/2.)
    else:
        return math.sqrt(np.array([i ** 2 for i in (np.array(a)-np.array(b))]).sum())


def eigen(a,b,x_axis,y_axis,z_axis):###"b"must be the index of the last third selected atom of helix 
    ###Given the beginning and endding index of alpha carbon atoms in a helix, caculate the eigenvalue, eigenvector, principal helical axis.
    sum_x=0; sum_y=0; sum_z=0;
    n= b-a
    for i in range(a,b):
        p_x=x_axis[i] + x_axis[i+2] - 2*x_axis[i+1]
        sum_x= p_x + sum_x
        p_y=y_axis[i] + y_axis[i+2] - 2*y_axis[i+1]
        sum_y= p_y + sum_y 
        p_z=z_axis[i] + z_axis[i+2] - 2*z_axis[i+1]
        sum_z= p_z + sum_z
        
    aver_sum_x=sum_x/n;aver_sum_y=sum_y/n;aver_sum_z=sum_z/n
    m_xy=0;  m_xz=0;  m_yz=0;  m_xx=0;  m_yy=0;  m_zz=0;
    sum_m_xy=0;  sum_m_xz=0;  sum_m_yz=0;  sum_m_xx=0;  sum_m_yy=0;  sum_m_zz=0;

    for i in range(a,b):
        m_xy=(x_axis[i] + x_axis[i+2] - 2*x_axis[i+1] - aver_sum_x)*(y_axis[i] + y_axis[i+2] - 2*y_axis[i+1] - aver_sum_y)
        sum_m_xy=m_xy + sum_m_xy
        m_xz=(x_axis[i] + x_axis[i+2] - 2*x_axis[i+1] - aver_sum_x)*(z_axis[i] + z_axis[i+2] - 2*z_axis[i+1] - aver_sum_z)
        sum_m_xz=m_xz + sum_m_xz
        m_yz=(y_axis[i] + y_axis[i+2] - 2*y_axis[i+1] - aver_sum_y)*(z_axis[i] + z_axis[i+2] - 2*z_axis[i+1] - aver_sum_z)
        sum_m_yz=m_yz + sum_m_yz
        m_xx=(x_axis[i] + x_axis[i+2] - 2*x_axis[i+1] - aver_sum_x)*(x_axis[i] + x_axis[i+2] - 2*x_axis[i+1] - aver_sum_x)
        sum_m_xx=m_xx + sum_m_xx
        m_yy=(y_axis[i] + y_axis[i+2] - 2*y_axis[i+1] - aver_sum_y)*(y_axis[i] + y_axis[i+2] - 2*y_axis[i+1] - aver_sum_y)
        sum_m_yy=m_yy + sum_m_yy
        m_zz=(z_axis[i] + z_axis[i+2] - 2*z_axis[i+1] - aver_sum_z)*(z_axis[i] + z_axis[i+2] - 2*z_axis[i+1] - aver_sum_z)
        sum_m_zz=m_zz + sum_m_zz

    m_11=sum_m_yy + sum_m_zz;m_12=-sum_m_xy;          m_13=-sum_m_xz
    m_21=-sum_m_xy;          m_22=sum_m_xx + sum_m_zz;m_23=-sum_m_yz
    m_31=-sum_m_xz;          m_32=-sum_m_yz;          m_33=sum_m_xx + sum_m_yy

    mat=np.array([[m_11,m_12,m_13],[m_21,m_22,m_23],[m_31,m_32,m_33]])
    e_value,e_vecters=LA.eig(mat)
    idx=e_value.argsort()[::-1]###Oder the eigen value from largest to the smallest.
    e_value=e_value[idx]
    e_vecters=e_vecters[:,idx]
    eigenvalue=e_value[0]; eigenvector=e_vecters[:,0]###Get the eigenvector of the largest eigenvalue
    hpa_x=-eigenvector[0];hpa_y=-eigenvector[1];hpa_z=-eigenvector[2]
    hpa=np.array([hpa_x, hpa_y, hpa_z])
    return eigenvalue,eigenvector,hpa

def vlen(v):
    ###Calculate the length of vector
    return np.sqrt(np.sum([ i*i for i in v ]))


def vangle(v1, v2):
    ###Calculat the angle beteween two vectors
    norm_v1 = v1 / vlen(v1)
    norm_v2 = v2 / vlen(v2)
    ang = np.degrees(np.arccos(round(np.vdot(norm_v1, norm_v2),5)))
    return ang

def vangle_mine(v1, v2):
    ###Calculat the angle beteween two vectors
    cos_radian_aa=v2[0]*v1[0] + v2[1]*v1[1] + v2[2]*v1[2]
    cos_radian_bb=(math.sqrt(v2[0]*v2[0] + v2[1]*v2[1] + v2[2]*v2[2]))*(math.sqrt(v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2]))
    cos_radian_2=cos_radian_aa/cos_radian_bb
    acos_radian_2=math.acos(cos_radian_2)
    angle=acos_radian_2*180/math.pi
    return angle

def Gram(vec1,vec2,vec3):
    ###Given three vectors(not orthogonalized), then keep one vectors unchanged, and to make other two vectors, in which the all the three vectors are othorgonal to each other.
    ###Use Gram-schmidt orthogonalization alogorithm. ###keep vec1 unchanged. 
    length_vec1 = math.sqrt(np.sum([ i*i for i in vec1 ]))
    dot1 = np.vdot(vec2,vec1)/(length_vec1*length_vec1)
    b = np.subtract(vec2,np.dot(dot1,vec1))
    length_vec2 = math.sqrt(np.sum([ i*i for i in b ]))
    dot2 = np.vdot(vec3,vec1)/(length_vec1*length_vec1)
    dot3 = np.vdot(vec3,b)/(length_vec2*length_vec2)
    c = np.subtract(vec3,np.dot(dot2,vec1))
    d = np.subtract(c,np.dot(dot3,b))
    return b,d


def project_points(x,y,z, a,b,c, aa,bb,cc):
    ###Projects the points with coordinates x, y, z onto the plane
    ###where a,b,c is the normal vector of the plane, aa,bb,cc is one point on the plane 
    normal_vector = np.array([a, b, c]) / np.sqrt(a*a + b*b + c*c)
    point_in_plane = np.array([aa,bb,cc])
    points = np.column_stack((x, y, z))
    points_from_point_in_plane = points - point_in_plane
    proj_onto_normal_vector = np.dot(points_from_point_in_plane,normal_vector)
    proj_onto_plane = (points_from_point_in_plane - proj_onto_normal_vector[:, None]*normal_vector)
    jj=point_in_plane + proj_onto_plane ##There are double brackets.
    jj_1=jj[0][0];jj_2=jj[0][1];jj_3=jj[0][2]##
    jj_array=np.array([jj_1,jj_2,jj_3])      ##Change the double brackets to one brackets
    return jj_array


if __name__ == "__main__":

    file = md.load(sys.argv[1]) ###load data
    
    ca_atoms = file.topology.select("name CA")
    x_axis = file.xyz[0,ca_atoms,0]
    y_axis = file.xyz[0,ca_atoms,1]
    z_axis = file.xyz[0,ca_atoms,2]
    
    np.savetxt("x_axis.dat",x_axis)
    np.savetxt("y_axis.dat",y_axis)
    np.savetxt("z_axis.dat",z_axis)
    
    
    ##############################################################
    ##############################################################Get the titling angle
    #n=10;
    a=10;b=24;
    print ("For helix","with index from", a,"to",b)
    
    hpa=eigen(10,24,x_axis,y_axis,z_axis)[2]
    norm_hpa=hpa / vlen(hpa)
    
    print ("principal helical axis", norm_hpa)
    print (" ")
    
    z=np.array([0,0,1])
    tilt=vangle(z,norm_hpa)
    
    if (tilt > 90):
        print ("tilting_angle", 180-tilt)
    else:
        print ("tilting_angle", tilt)
    ##############################################################
    ##############################################################Get the rotation angle method one
    c=(a+b)/2
    rs_x=x_axis[c];rs_y=y_axis[c];rs_z=z_axis[c] ###Select one atom at the helix as the point the make a perpendicular vector
    hpa_x=hpa[0];hpa_y=hpa[1];hpa_z=hpa[2]
    t_value=(rs_x*hpa_x + rs_y*hpa_y + rs_z*hpa_z)/(hpa_x*hpa_x + hpa_y*hpa_y + hpa_z*hpa_z)
    
    rs_vector_x=rs_x - t_value*hpa_x;rs_vector_y=rs_y - t_value*hpa_y;rs_vector_z=rs_z - t_value*hpa_z;
    rs_vector=np.array([rs_vector_x, rs_vector_y, rs_vector_z]) ###The made perpendicular vector
    
    print ("test", np.dot(rs_vector,hpa))
    #zp=project_points(rs_x,rs_y,rs_z,hpa[0],hpa[1],hpa[2],rs_x,rs_y,rs_z)
    #zp=project_points(0,0,1,norm_hpa[0],norm_hpa[1],norm_hpa[2],rs_x,rs_y,rs_z)
    zp=project_points(0,0,1,norm_hpa[0],norm_hpa[1],norm_hpa[2],rs_vector_x,rs_vector_y,rs_vector_z)
    
    rot=vangle_mine(rs_vector,zp)
    
    #############################################################Get the rotation angle method two
    c=(a+b)/2
    d=15
    g=14
    
    r_c=np.array([x_axis[c]-x_axis[g], y_axis[c]-y_axis[g], z_axis[c]-z_axis[g]])
    r_d=np.array([x_axis[d]-x_axis[g], y_axis[d]-y_axis[g], z_axis[d]-z_axis[g]])
    
    r_vec_1=Gram(hpa,r_c,r_d)[0]
    r_vec_2=Gram(hpa,r_c,r_d)[1]
    
    print (r_vec_1)
    zp=project_points(0,0,1,norm_hpa[0],norm_hpa[1],norm_hpa[2],r_vec_1[0],r_vec_1[1],r_vec_1[2])
    rot_2=vangle_mine(r_vec_1,zp)
    
    ##############################################################Get the rotation angle method three
    PROJ_K = z - np.vdot(z, norm_hpa)*norm_hpa
    rot_3 = vangle(PROJ_K, r_vec_1)
    
    dot = np.dot(PROJ_K, r_vec_2)
    
    sign = math.fabs(dot)/dot
    
    print ("rotated_angle", rot_3*sign)
    
    
    ##############################################################
    #y=np.array([0,1,0])
    #test_1=vangle_mine(r_vec_1,y)
    #print "y_angle", test_1
    ##############################################################
    ##############################################################
    ##############################################################
